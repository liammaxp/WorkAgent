from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend import github_evidence_materializer as materializer
from backend import github_evidence_chunks
from backend import github_raw_storage
from backend import project_retrieval_v2


PRIVATE = "diff --git a/private.py b/private.py\nSECRET_API_KEY=example-secret\n"


def memory(*projects):
    return {"projects": list(projects)}


def project(project_id="ProjectA", repo="owner/repo-a"):
    return {"project_id": project_id, "repository": repo}


def context(*, repo="owner/repo-a", project_id=None, readme="# Project\nRetrieval evidence.", patch="@@ def retrieve():\n+validation\n"):
    value = {
        "repository": repo, "latest_commit_sha": "abc123", "readme": readme,
        "contribution_evidence": [{"commits": [{
            "sha": "abc123", "message": "change retrieval",
            "file_changes": [{"filename": "backend/retrieval.py", "patch": patch}],
        }]}],
    }
    if project_id is not None:
        value["project_id"] = project_id
    return value


def paths(tmp_path):
    return {
        "raw_source_path": tmp_path / "github_raw_sources.jsonl",
        "chunk_path": tmp_path / "github_evidence_chunks.jsonl",
        "manifest_path": tmp_path / "github_evidence_materialization.json",
    }


@pytest.mark.parametrize("value", (None, "", "0", "false", "invalid"))
def test_materialization_defaults_off_and_invalid_values_fail_closed(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(materializer.GITHUB_EVIDENCE_MATERIALIZATION_FLAG, raising=False)
    else:
        monkeypatch.setenv(materializer.GITHUB_EVIDENCE_MATERIALIZATION_FLAG, value)
    blocked_input = tmp_path / "must-not-read.json"
    result = materializer.materialize_saved_github_evidence(
        saved_context_path=blocked_input, project_memory_path=blocked_input, **paths(tmp_path)
    )
    assert result["status"] == "disabled"
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv(materializer.GITHUB_EVIDENCE_MATERIALIZATION_FLAG, "1")
    invalid = materializer.materialize_saved_github_evidence(
        feature_enabled="yes", saved_context_path=blocked_input, project_memory_path=blocked_input,
        **paths(tmp_path),
    )
    assert invalid["status"] == "disabled"


def test_flags_are_independent(monkeypatch, tmp_path):
    monkeypatch.setenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "1")
    monkeypatch.delenv(materializer.GITHUB_EVIDENCE_MATERIALIZATION_FLAG, raising=False)
    assert not materializer.is_github_evidence_materialization_enabled()
    assert materializer.materialize_saved_github_evidence(**paths(tmp_path))["status"] == "disabled"
    monkeypatch.setenv(materializer.GITHUB_EVIDENCE_MATERIALIZATION_FLAG, "1")
    monkeypatch.delenv(project_retrieval_v2.GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, raising=False)
    assert materializer.is_github_evidence_materialization_enabled()
    assert not project_retrieval_v2.is_github_evidence_retrieval_v2_enabled()


def test_saved_context_adapter_builds_readme_patch_and_deduplicates():
    mapping = materializer.build_authoritative_repository_project_mapping(memory(project()))
    saved = {"repositories": {
        "owner/repo-a": {"context": context()},
        "OWNER/REPO-A": {"context": context()},
    }}
    records, stats = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=saved, authoritative_mapping=mapping
    )
    assert len(records) == 2
    assert {record["source_type"] for record in records} == {"readme", "commit_patch"}
    assert {record["path"] for record in records} == {"README.md", "backend/retrieval.py"}
    assert all(record["project_id"] == "ProjectA" for record in records)
    assert stats["conflict_count"] == stats["unresolved_identity_count"] == 0


def test_missing_identity_resolves_only_by_unique_mapping_and_conflict_skips():
    unique = materializer.build_authoritative_repository_project_mapping(memory(project()))
    records, _ = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=[context()], authoritative_mapping=unique
    )
    assert records and {record["project_id"] for record in records} == {"ProjectA"}
    conflict = materializer.build_authoritative_repository_project_mapping(memory(
        project("ProjectA"), project("ProjectB"),
    ))
    blocked, stats = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=[context()], authoritative_mapping=conflict
    )
    assert blocked == [] and stats["conflict_count"] == 1
    unresolved, stats = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=[context(repo="owner/unknown")], authoritative_mapping=unique
    )
    assert unresolved == [] and stats["unresolved_identity_count"] == 1


def test_empty_and_unsupported_saved_content_skips_safely():
    mapping = materializer.build_authoritative_repository_project_mapping(memory(project()))
    records, stats = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=[context(readme="", patch=""), {"repository": "owner/repo-a", "readme": 123}],
        authoritative_mapping=mapping,
    )
    assert records == []
    assert stats["source_records_accepted"] == 0
    missing_repo, stats = materializer.adapt_saved_github_contexts_to_raw_sources(
        saved_context=[{"project_id": "ProjectA", "readme": "private"}],
        authoritative_mapping=mapping,
    )
    assert missing_repo == [] and stats["unresolved_identity_count"] == 1


def test_materialization_creates_stable_raw_chunks_manifest_and_is_idempotent(tmp_path):
    kwargs = {
        "feature_enabled": True, "saved_context": [context()],
        "project_memory": memory(project()), **paths(tmp_path),
    }
    first = materializer.materialize_saved_github_evidence(**kwargs)
    before = {name: path.read_bytes() for name, path in paths(tmp_path).items()}
    second = materializer.materialize_saved_github_evidence(**kwargs)
    after = {name: path.read_bytes() for name, path in paths(tmp_path).items()}
    assert first["status"] == "created" and second["status"] == "unchanged"
    assert before == after
    raw = github_raw_storage.load_github_raw_source_records(paths(tmp_path)["raw_source_path"])
    chunks = github_evidence_chunks.load_github_evidence_chunk_records(paths(tmp_path)["chunk_path"])
    manifest = json.loads(paths(tmp_path)["manifest_path"].read_text(encoding="utf-8"))
    assert len(raw) == first["raw_source_count"] == 2
    assert len(chunks) == first["chunk_count"]
    assert len({record["source_id"] for record in raw}) == len(raw)
    assert len({record["chunk_id"] for record in chunks}) == len(chunks)
    assert manifest["status"] == "ready"
    assert manifest["raw_artifact_hash"] == hashlib.sha256(before["raw_source_path"]).hexdigest()


def test_input_reordering_is_logically_and_byte_stable(tmp_path):
    first_paths = paths(tmp_path / "first")
    second_paths = paths(tmp_path / "second")
    contexts = [context(), context(project_id="ProjectA", readme="# Other", patch="@@ class Cache:\n+cache")]
    first = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=contexts, project_memory=memory(project()), **first_paths
    )
    second = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=list(reversed(contexts)), project_memory=memory(project()), **second_paths
    )
    assert first["content_hash"] == second["content_hash"]
    for key in first_paths:
        assert first_paths[key].read_bytes() == second_paths[key].read_bytes()


def test_modified_input_updates_and_malformed_input_preserves_previous_pair(tmp_path):
    target_paths = paths(tmp_path)
    base = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context()], project_memory=memory(project()), **target_paths
    )
    before = {key: path.read_bytes() for key, path in target_paths.items()}
    updated = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context(readme="# Changed")],
        project_memory=memory(project()), **target_paths,
    )
    assert base["status"] == "created" and updated["status"] == "updated"
    valid = {key: path.read_bytes() for key, path in target_paths.items()}
    blocked = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context(repo="owner/unknown")],
        project_memory=memory(project()), **target_paths,
    )
    assert blocked["status"] == "blocked"
    assert {key: path.read_bytes() for key, path in target_paths.items()} == valid
    assert before != valid


@pytest.mark.parametrize("failure_call", (1, 2, 3))
def test_atomic_replacement_failure_restores_previous_artifact_trio(tmp_path, failure_call):
    target_paths = paths(tmp_path)
    materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context()], project_memory=memory(project()), **target_paths
    )
    before = {key: path.read_bytes() for key, path in target_paths.items()}
    calls = 0
    def failing_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("simulated replacement failure")
        return materializer.os.replace(source, target)
    result = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context(readme="# Modified")],
        project_memory=memory(project()), replace=failing_replace, **target_paths,
    )
    assert result["status"] == "error"
    assert {key: path.read_bytes() for key, path in target_paths.items()} == before


def test_limits_report_partial_without_claiming_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(materializer, "MAX_RAW_SOURCES_TOTAL", 1)
    result = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context()], project_memory=memory(project()), **paths(tmp_path)
    )
    assert result["status"] == "partial"
    manifest = json.loads(paths(tmp_path)["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert result["warnings"]


def test_invalid_limits_fail_closed_without_artifact_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(materializer, "MAX_RAW_SOURCES_TOTAL", 0)
    result = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context()], project_memory=memory(project()), **paths(tmp_path)
    )
    assert result["status"] == "error"
    assert result["errors"] == ["invalid_materialization_limits"]
    assert list(tmp_path.iterdir()) == []


def test_results_are_redacted_while_backend_artifact_may_hold_private_source(tmp_path):
    result = materializer.materialize_saved_github_evidence(
        feature_enabled=True, saved_context=[context(readme=PRIVATE, patch=PRIVATE)],
        project_memory=memory(project()), **paths(tmp_path),
    )
    serialized = json.dumps(result, sort_keys=True)
    assert "diff --git" not in serialized and "example-secret" not in serialized
    stored = github_raw_storage.load_github_raw_source_records(paths(tmp_path)["raw_source_path"])
    assert any(record["raw_text"] == PRIVATE for record in stored)
    assert "raw_text" not in result and "text" not in result


def test_materializer_has_no_network_chroma_api_llm_or_capability_dependencies():
    source = Path(materializer.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "import requests", "urlopen(", "chromadb", "openai", "api_server", "projectcapabilityfact",
        "project_capability_reader", "project_capability_pipeline", "project_capability_backfill",
    ):
        assert forbidden not in source
    assert project_retrieval_v2.retrieve_evidence_for_project_v2({"project_id": "ProjectA"}) == []
