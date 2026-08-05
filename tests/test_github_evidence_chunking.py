from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend import github_evidence_chunks as chunks
from backend import github_raw_storage
from backend import project_capability_reader
from backend import project_retrieval_v2


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SENTINEL = "FULL_PRIVATE_FILE_CONTENT"


def raw_record(**changes):
    values = {
        "project_id": "WorkAgent",
        "repo": "WorkAgent",
        "source_type": "file_snapshot",
        "path": "backend/retrieval.py",
        "commit_sha": "abc123",
        "raw_text": "def retrieve_evidence():\n    return validateCache()\n",
        "metadata": {"language": "Python", "raw_body": PRIVATE_SENTINEL},
    }
    values.update(changes)
    return github_raw_storage.build_github_raw_source_record(**values)


def test_chunk_record_is_deterministic_and_preserves_source_fields():
    raw = raw_record()
    values = {
        "source_id": raw["source_id"],
        "project_id": raw["project_id"],
        "repo": raw["repo"],
        "source_type": raw["source_type"],
        "chunk_type": "file_section",
        "path": raw["path"],
        "commit_sha": raw["commit_sha"],
        "text": raw["raw_text"],
        "raw_hash": raw["raw_hash"],
    }
    first = chunks.build_github_evidence_chunk_record(**values)
    second = chunks.build_github_evidence_chunk_record(**values)
    assert first == second
    assert first["chunk_id"].startswith("chk_")
    assert first["source_id"] == raw["source_id"]
    assert first["raw_hash"] == raw["raw_hash"]
    assert first["text_hash"] == hashlib.sha256(raw["raw_text"].encode()).hexdigest()
    assert first["text_chars"] == len(raw["raw_text"])


def test_empty_and_optional_chunk_fields_are_safe():
    record = chunks.build_github_evidence_chunk_record(text="")
    assert record["text"] == ""
    assert record["text_chars"] == 0
    assert record["keywords"] == record["technical_tags"] == []
    assert chunks.build_github_evidence_chunks_from_raw_source(raw_record(raw_text="")) == []


@pytest.mark.parametrize("value", (123, b"text", ["text"], {"text": "value"}))
def test_non_string_chunk_text_is_rejected(value):
    with pytest.raises(TypeError, match="text"):
        chunks.build_github_evidence_chunk_record(text=value)


def test_commit_patch_splits_files_and_hunks_deterministically():
    patch = (
        "diff --git a/backend/a.py b/backend/a.py\n"
        "--- a/backend/a.py\n+++ b/backend/a.py\n"
        "@@ -1 +1,2 @@ def retrieve_evidence():\n-old\n+validated retrieval evidence\n"
        "@@ -8 +9 @@ class CacheStore:\n-old cache\n+new cache\n"
        "diff --git a/backend/b.py b/backend/b.py\n"
        "--- a/backend/b.py\n+++ b/backend/b.py\n"
        "@@ -1 +1 @@ def rerankResults():\n-old\n+validation rerank SQLite merge\n"
    )
    raw = raw_record(source_type="commit_patch", path="", raw_text=patch)
    first = chunks.build_github_evidence_chunks_from_raw_source(raw)
    second = chunks.build_github_evidence_chunks_from_raw_source(raw)
    assert first == second
    assert len(first) == 3
    assert {item["path"] for item in first} == {"backend/a.py", "backend/b.py"}
    assert all(item["chunk_type"] == "diff_hunk" for item in first)
    assert all(item["commit_sha"] == "abc123" for item in first)
    assert all(len(item["text"]) <= chunks.MAX_CHUNK_CHARS for item in first)
    assert first[0]["symbol"] == "retrieve_evidence"


def test_readme_headings_create_deterministic_sections():
    raw = raw_record(
        source_type="readme",
        path="README.md",
        raw_text="# Retrieval\nEvidence overview.\n## Validation\nQuality gate details.\n",
    )
    result = chunks.build_github_evidence_chunks_from_raw_source(raw)
    assert len(result) == 2
    assert all(item["chunk_type"] == "readme_section" for item in result)
    assert all(item["path"] == "README.md" and item["repo"] == "WorkAgent" for item in result)
    assert result == chunks.build_github_evidence_chunks_from_raw_source(raw)


def test_large_fallback_is_bounded_and_enforces_max_chunks():
    raw = raw_record(source_type="unknown", raw_text=("x" * chunks.MAX_CHUNK_CHARS + "\n") * 80)
    result = chunks.build_github_evidence_chunks_from_raw_source(raw)
    assert len(result) == chunks.MAX_CHUNKS_PER_SOURCE
    assert all(0 < len(item["text"]) <= chunks.MAX_CHUNK_CHARS for item in result)
    assert all(item["chunk_type"] == "text_window" for item in result)


def test_file_snapshot_and_log_use_structure_aware_chunks():
    source = "def snake_case():\n pass\nclass PascalCase:\n pass\nfunction camelCase() {}\n"
    file_chunks = chunks.build_github_evidence_chunks_from_raw_source(raw_record(raw_text=source))
    log_chunks = chunks.build_github_evidence_chunks_from_raw_source(
        raw_record(source_type="log", raw_text="2026-01-01 started\n detail\n2026-01-02 stopped\n")
    )
    assert [item["symbol"] for item in file_chunks] == ["snake_case", "PascalCase", "camelCase"]
    assert all(item["chunk_type"] == "file_section" for item in file_chunks)
    assert len(log_chunks) == 2
    assert all(item["chunk_type"] == "log_entry" for item in log_chunks)


def test_keyword_symbol_and_technical_tag_extraction_is_bounded_and_stable():
    text = "def snake_case(): camelCase PascalCase retrieval validation LaTeX SQLite cache rerank diff merge evidence Chroma fallback quality gate"
    assert chunks.extract_github_evidence_symbol(text) == "snake_case"
    keywords = chunks.extract_github_evidence_keywords(text)
    tags = chunks.extract_github_evidence_technical_tags(text)
    assert {"snake_case", "camelcase", "pascalcase"} <= set(keywords)
    assert {"retrieval", "validation", "latex", "sqlite", "cache", "rerank", "diff", "merge", "evidence"} <= set(tags)
    assert len(keywords) <= chunks.MAX_KEYWORDS
    assert tags == chunks.extract_github_evidence_technical_tags(text)


def test_chunk_jsonl_round_trip_preserves_backend_text(tmp_path):
    artifact = tmp_path / "github_evidence_chunks.jsonl"
    records = chunks.build_github_evidence_chunks_from_raw_source(raw_record())
    chunks.append_github_evidence_chunk_record(records[0], artifact)
    chunks.append_github_evidence_chunk_record(records[0], artifact)
    assert chunks.load_github_evidence_chunk_records(artifact) == records
    assert chunks.load_github_evidence_chunk_records(artifact)[0]["text"] == records[0]["text"]


@pytest.mark.parametrize("line", ("not-json\n", "[]\n"))
def test_malformed_chunk_jsonl_fails_safely(tmp_path, line):
    artifact = tmp_path / "github_evidence_chunks.jsonl"
    artifact.write_text(line, encoding="utf-8")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        chunks.load_github_evidence_chunk_records(artifact)


def test_chunk_redaction_excludes_text_and_sensitive_metadata():
    record = chunks.build_github_evidence_chunks_from_raw_source(
        raw_record(raw_text=f"def retrieve_evidence():\n    return '{PRIVATE_SENTINEL}'\n")
    )[0]
    safe = chunks.redact_github_evidence_chunk_record(record)
    serialized = json.dumps(safe, sort_keys=True)
    assert "text" not in safe
    assert "raw_text" not in safe
    assert PRIVATE_SENTINEL not in serialized
    assert "diff --git" not in serialized
    assert safe["text_available"] is True
    assert safe["text_hash"] == record["text_hash"]
    assert safe["text_chars"] == record["text_chars"]
    assert safe["metadata"] == {"language": "Python"}


def test_redaction_safe_labels_cannot_smuggle_source_content():
    record = chunks.build_github_evidence_chunk_record(
        text="bounded",
        symbol="diff --git private patch",
        keywords=["retrieval", "diff --git private patch"],
        technical_tags=["validation", "FULL PRIVATE FILE CONTENT"],
    )
    serialized = json.dumps(chunks.redact_github_evidence_chunk_record(record), sort_keys=True)
    assert "diff --git" not in serialized
    assert "FULL PRIVATE FILE CONTENT" not in serialized
    assert record["keywords"] == ["retrieval"]
    assert record["technical_tags"] == ["validation"]


def test_retrieval_scaffold_remains_empty_and_capability_reader_remains_off(monkeypatch):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    monkeypatch.delenv(project_capability_reader.PROJECT_CAPABILITY_MEMORY_FLAG, raising=False)
    monkeypatch.setattr(
        project_capability_reader,
        "read_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("chunking must not invoke capability reader"),
    )
    result = project_retrieval_v2.retrieve_evidence_for_project_v2(
        {"raw_text": PRIVATE_SENTINEL, "text": PRIVATE_SENTINEL}
    )
    assert result == []
    assert PRIVATE_SENTINEL not in json.dumps(result)
    assert project_capability_reader.is_project_capability_memory_enabled() is False


def test_chunking_has_no_capability_lifecycle_dependencies():
    source = Path(chunks.__file__).read_text(encoding="utf-8")
    forbidden = (
        "ProjectCapabilityFact",
        "project_capability_reader",
        "project_capability_pipeline",
        "project_capability_backfill",
        "rebuild",
    )
    assert all(term not in source for term in forbidden)


def test_no_frontend_or_real_information_artifact_changes():
    source = (ROOT / "backend" / "github_evidence_chunks.py").read_text(encoding="utf-8")
    assert "frontend/" not in source and "information/" not in source
