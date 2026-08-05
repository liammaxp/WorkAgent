from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import api_server
from backend import project_retrieval_v2
from backend import project_capability_reader


@pytest.mark.parametrize("value", (None, "", "0", "false", "no", "off", "invalid"))
def test_retrieval_v2_defaults_off_and_preserves_legacy_routing(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, raising=False)
    else:
        monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, value)

    project = {"project_id": "safe-project"}
    expected = [{"source": "legacy", "text": "existing shaped result"}]
    monkeypatch.setattr(api_server, "retrieve_evidence_for_project", lambda item: expected if item is project else [])
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda _item: pytest.fail("v2 must not run while disabled"),
    )

    assert project_retrieval_v2.is_github_evidence_retrieval_v2_enabled() is False
    assert api_server.retrieve_evidence_for_project_for_resume(project) is expected


def test_retrieval_v2_flag_on_routes_to_v2_only(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    expected = [{"project_id": "safe-project", "chunk_id": "chk_safe"}]
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda _item: pytest.fail("legacy retrieval must not run while v2 is enabled"),
    )
    monkeypatch.setattr(
        project_retrieval_v2,
        "retrieve_evidence_for_project_v2",
        lambda _item, **_kwargs: expected,
    )

    result = api_server.retrieve_evidence_for_project_for_resume({"project_id": "safe-project"})

    assert project_retrieval_v2.is_github_evidence_retrieval_v2_enabled() is True
    assert result is expected


def test_retrieval_v2_empty_result_never_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    monkeypatch.setattr(
        api_server,
        "retrieve_evidence_for_project",
        lambda _item: pytest.fail("legacy fallback is forbidden"),
    )
    monkeypatch.setattr(project_retrieval_v2, "retrieve_evidence_for_project_v2", lambda _item, **_kwargs: [])

    assert api_server.retrieve_evidence_for_project_for_resume({"project_id": "safe-project"}) == []


def test_retrieval_v2_does_not_enable_or_invoke_capability_reader(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    monkeypatch.delenv(project_capability_reader.PROJECT_CAPABILITY_MEMORY_FLAG, raising=False)
    monkeypatch.setattr(
        project_capability_reader,
        "read_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("retrieval v2 must not invoke the capability reader"),
    )

    result = api_server.retrieve_evidence_for_project_for_resume({"project_id": "zero-fact-project"})

    assert result == []
    assert project_capability_reader.is_project_capability_memory_enabled() is False


def test_retrieval_v2_module_has_no_capability_lifecycle_or_schema_dependencies():
    source = Path(project_retrieval_v2.__file__).read_text(encoding="utf-8")
    forbidden = (
        "project_capability_reader",
        "project_capability_pipeline",
        "project_capability_backfill",
        "ProjectCapabilityFact",
        "build_project_capability",
        "rebuild",
    )
    assert all(name not in source for name in forbidden)


def test_retrieval_v2_change_does_not_modify_frontend_files():
    source = (ROOT / "backend" / "project_retrieval_v2.py").read_text(encoding="utf-8")
    assert "frontend/" not in source and "frontend\\" not in source
