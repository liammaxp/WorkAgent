from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend import github_raw_storage
from backend import project_capability_reader
from backend import project_retrieval_v2


ROOT = Path(__file__).resolve().parents[1]
RAW_SENTINEL = "diff --git a/private.py b/private.py\nFULL_FILE_CONTENT_SENTINEL"


def _record(**changes):
    values = {
        "project_id": "WorkAgent",
        "repo": "WorkAgent",
        "source_type": "commit_patch",
        "path": "backend/api_server.py",
        "commit_sha": "abc123",
        "raw_text": RAW_SENTINEL,
        "created_at": "2026-08-01T00:00:00Z",
        "metadata": {"language": "Python", "patch": RAW_SENTINEL, "body": RAW_SENTINEL},
    }
    values.update(changes)
    return github_raw_storage.build_github_raw_source_record(**values)


def test_raw_record_is_deterministic_and_internally_consistent():
    first = _record()
    second = _record()
    assert first == second
    assert first["source_id"].startswith("raw_")
    assert first["raw_hash"] == hashlib.sha256(RAW_SENTINEL.encode("utf-8")).hexdigest()
    assert first["raw_chars"] == len(RAW_SENTINEL)


def test_optional_fields_and_empty_raw_text_are_safe():
    record = github_raw_storage.build_github_raw_source_record(raw_text="")
    assert record["project_id"] == record["repo"] == record["path"] == ""
    assert record["source_type"] == "unknown"
    assert record["raw_chars"] == 0
    assert record["raw_hash"] == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("value", (123, b"bytes", ["text"], {"text": "value"}))
def test_non_string_raw_text_is_rejected(value):
    with pytest.raises(TypeError, match="raw_text"):
        github_raw_storage.build_github_raw_source_record(raw_text=value)


def test_jsonl_round_trip_preserves_backend_raw_text_and_is_idempotent(tmp_path):
    path = tmp_path / "github_raw_sources.jsonl"
    first = _record()
    second = _record(path="README.md", source_type="readme", raw_text="private readme")
    github_raw_storage.append_github_raw_source_record(first, path)
    github_raw_storage.append_github_raw_source_record(second, path)
    github_raw_storage.append_github_raw_source_record(first, path)

    loaded = github_raw_storage.load_github_raw_source_records(path)
    assert loaded == [first, second]
    assert loaded[0]["raw_text"] == RAW_SENTINEL
    assert path.parent == tmp_path


@pytest.mark.parametrize("line", ("not json\n", "[]\n"))
def test_malformed_jsonl_fails_safely(tmp_path, line):
    path = tmp_path / "github_raw_sources.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        github_raw_storage.load_github_raw_source_records(path)


def test_redaction_excludes_raw_and_large_content_but_keeps_safe_fields():
    safe = github_raw_storage.redact_github_raw_source_record(_record())
    serialized = json.dumps(safe, sort_keys=True)
    assert "raw_text" not in safe
    assert "patch" not in safe.get("metadata", {})
    assert "body" not in safe.get("metadata", {})
    assert RAW_SENTINEL not in serialized
    assert "diff --git" not in serialized
    assert "FULL_FILE_CONTENT_SENTINEL" not in serialized
    assert safe["raw_available"] is True
    assert safe["raw_hash"] == hashlib.sha256(RAW_SENTINEL.encode()).hexdigest()
    assert safe["raw_chars"] == len(RAW_SENTINEL)
    assert safe["metadata"] == {"language": "Python"}


def test_v2_scaffold_exposes_no_raw_content(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    result = project_retrieval_v2.retrieve_evidence_for_project_v2(
        {"project_id": "WorkAgent", "raw_text": RAW_SENTINEL}
    )
    assert result == []
    assert RAW_SENTINEL not in json.dumps(result)


def test_capability_reader_stays_default_off_with_zero_facts(monkeypatch):
    monkeypatch.delenv(project_capability_reader.PROJECT_CAPABILITY_MEMORY_FLAG, raising=False)
    monkeypatch.setattr(
        project_capability_reader,
        "read_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("raw storage must not invoke capability reader"),
    )
    assert project_capability_reader.is_project_capability_memory_enabled() is False
    assert github_raw_storage.redact_github_raw_source_record(_record())["raw_available"] is True


def test_raw_storage_has_no_capability_lifecycle_dependencies():
    source = Path(github_raw_storage.__file__).read_text(encoding="utf-8")
    forbidden = (
        "ProjectCapabilityFact",
        "project_capability_reader",
        "project_capability_pipeline",
        "project_capability_backfill",
        "rebuild",
        "fallback",
    )
    assert all(term not in source for term in forbidden)


def test_no_frontend_or_real_information_artifact_changes():
    source = (ROOT / "backend" / "github_raw_storage.py").read_text(encoding="utf-8")
    assert "frontend/" not in source and "information/" not in source
