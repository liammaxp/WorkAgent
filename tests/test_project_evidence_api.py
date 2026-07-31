import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import evidence_memory  # noqa: E402
import project_change_pipeline  # noqa: E402


RAW_KEYS = {"raw_text", "patch_text", "hunk_text", "added_lines", "removed_lines", "content"}


def test_api_server_imports_from_supported_backend_working_directory():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", "import api_server; print(api_server.app.title)"],
        cwd=ROOT / "backend",
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "WorkAgent API"


def keys(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key).lower())
            found.update(keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(keys(item))
    return found


def test_semantic_status_never_returns_raw_github_content():
    client = TestClient(api_server.app)
    with (
        patch.object(api_server, "github_context_status_v2_enabled", return_value=True),
        patch.object(api_server, "get_github_context_status_v2", return_value={
            "enabled": True, "saved": True, "repo_count": 1, "record_count": 2,
            "raw_chars": 999, "errors": [],
        }),
        patch.object(api_server.project_change_pipeline, "get_project_change_health", return_value={
            "enabled": True, "status": "ready", "schema_version": "project_change_memory.v1",
            "memory_exists": True, "memory_readable": True, "project_count": 1,
            "raw_change_summary_count": 1, "evidence_card_count": 1,
            "capability_fact_count": 1, "issues": [],
        }),
    ):
        response = client.get("/api/project-evidence/status")
    assert response.status_code == 200
    assert not (keys(response.json()) & RAW_KEYS)


def test_semantic_preview_is_metadata_only_and_bounded(monkeypatch):
    monkeypatch.delenv("USE_PROJECT_EVIDENCE_MEMORY", raising=False)
    payload = TestClient(api_server.app).get("/api/project-evidence/preview?limit=999").json()
    assert payload["sample_limit"] == api_server.semantic_evidence_pipeline.MAX_INSPECT_SAMPLE_LIMIT
    assert not (keys(payload) & RAW_KEYS)


def test_semantic_raw_access_is_explicit_and_bounded():
    client = TestClient(api_server.app)
    with patch.object(api_server, "github_context_status_v2_enabled", return_value=True):
        missing = client.get("/api/project-evidence/raw?max_chars=999999")
    assert missing.status_code == 400
    with (
        patch.object(api_server, "github_context_status_v2_enabled", return_value=True),
        patch.object(api_server, "get_github_context_raw_v2", return_value={
            "enabled": True, "source_id": "source-a", "max_chars": 30_000,
            "raw_text": "bounded", "returned_chars": 7, "truncated": True, "errors": [],
        }) as read_raw,
    ):
        response = client.get("/api/project-evidence/raw?source_id=source-a&max_chars=999999")
    assert response.status_code == 200
    read_raw.assert_called_once_with(source_id="source-a", max_chars=api_server.GITHUB_CONTEXT_RAW_MAX_CHARS)


def test_canonical_github_raw_patch_inputs_are_deduped(tmp_path, monkeypatch):
    monkeypatch.setenv(evidence_memory.GITHUB_EVIDENCE_MEMORY_DIR_ENV, str(tmp_path))
    common = {
        "project_id": "sample", "repo": "owner/sample", "source_type": "commit_patch",
        "path": "backend/app.py", "commit_sha": "abc", "raw_text": "@@ -1 +1 @@\n-old\n+new",
    }
    evidence_memory.upsert_github_raw_source({**common, "source_id": "source-a"})
    evidence_memory.upsert_github_raw_source({**common, "source_id": "source-b"})
    inputs, skipped = project_change_pipeline.load_github_raw_sources_for_project_change_memory()
    assert len(inputs) == 1
    assert inputs[0].file_path == "backend/app.py"
    assert skipped == []


def test_manual_evidence_controls_are_development_only():
    source = (ROOT / "frontend" / "src" / "pages" / "GitHubContext.jsx").read_text(encoding="utf-8")
    assert "const showEvidenceDebug = import.meta.env.DEV" in source
    assert '{showEvidenceDebug && <section className="card evidence-pipeline-panel">' in source
