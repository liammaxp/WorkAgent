from __future__ import annotations

import json
from pathlib import Path
import threading

from backend import github_evidence_preparation_service as service
from backend import github_evidence_materializer as materializer
from backend import project_repository_identity as identity


def project_memory(*project_ids):
    return {"projects": [{"project_id": value} for value in project_ids]}


def saved_context(repository="owner/repo", body="saved evidence"):
    return {"repositories": {repository: {
        "repository": repository,
        "context": {"repository": repository, "readme": body},
    }}}


def authority(memory, *links):
    return identity.build_project_repository_identity_authority(
        project_memory=memory, user_confirmed_links=list(links),
    )


def link(project_id="A", repository="owner/repo"):
    return {"project_id": project_id, "repository": repository, "confirmed": True}


def vector(repository="owner/repo"):
    return {"id": "vector-one", "metadata": {"repository": repository}}


def paths(tmp_path):
    return {
        "raw_source_path": tmp_path / "raw.jsonl",
        "chunk_path": tmp_path / "chunks.jsonl",
        "manifest_path": tmp_path / "manifest.json",
    }


def ready_inputs(tmp_path, body="saved evidence"):
    memory = project_memory("A")
    return {
        "feature_enabled": True, "saved_context": saved_context(body=body),
        "project_memory": memory, "identity_authority": authority(memory, link()),
        "confirmations": {"confirmations": [link()]}, "vector_records": [vector()],
        **paths(tmp_path),
    }


def assert_product_safe(result):
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "raw_source_count", "chunk_count", "vector_record_count", "content_hash",
        "artifact_hash", "raw_text", "diff --git", "BEGIN PRIVATE KEY", "C:\\",
    ):
        assert forbidden not in serialized


def test_disabled_and_invalid_feature_values_fail_closed_without_io(tmp_path):
    for value in (False, "true", 1):
        target = tmp_path / str(value)
        result = service.get_github_evidence_preparation_status(
            feature_enabled=value, saved_context=saved_context(), project_memory=project_memory("A"),
            identity_authority=None, confirmations={}, vector_records=[vector()],
            raw_source_path=target / "raw", chunk_path=target / "chunks", manifest_path=target / "manifest",
        )
        assert result["status"] == "disabled"
        assert result["can_prepare"] is False
        assert not target.exists()
        assert_product_safe(result)


def test_unresolved_mapping_conflict_and_missing_context_block_safely(tmp_path):
    memory = project_memory("A", "B")
    missing = service.get_github_evidence_preparation_status(
        feature_enabled=True, saved_context=saved_context(), project_memory=memory,
        identity_authority=None, confirmations={}, vector_records=[vector()], **paths(tmp_path),
    )
    assert missing["status"] == "mapping_required"
    assert missing["requires_repository_mapping"] is True
    conflict_authority = authority(memory, link("A"), link("B"))
    conflict = service.get_github_evidence_preparation_status(
        feature_enabled=True, saved_context=saved_context(), project_memory=memory,
        identity_authority=conflict_authority, confirmations={}, vector_records=[vector()], **paths(tmp_path),
    )
    assert conflict["status"] == "blocked" and conflict["conflict_count"] > 0
    unavailable = service.get_github_evidence_preparation_status(
        feature_enabled=True, saved_context={}, project_memory=project_memory("A"),
        identity_authority=authority(project_memory("A"), link()), confirmations={},
        vector_records=[], **paths(tmp_path),
    )
    assert unavailable["status"] == "blocked"


def test_complete_mapping_is_ready_to_prepare_and_get_never_materializes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service, "materialize_saved_github_evidence",
        lambda **_: (_ for _ in ()).throw(AssertionError("GET must not materialize")),
    )
    inputs = ready_inputs(tmp_path)
    first = service.get_github_evidence_preparation_status(**inputs)
    second = service.get_github_evidence_preparation_status(**inputs)
    assert first == second
    assert first["status"] == "ready_to_prepare" and first["can_prepare"] is True
    assert not any(path.exists() for path in paths(tmp_path).values())


def test_real_materializer_created_unchanged_updated_and_prepared_status(tmp_path):
    inputs = ready_inputs(tmp_path)
    first = service.run_github_evidence_preparation(confirmed=True, **inputs)
    before = {key: path.read_bytes() for key, path in paths(tmp_path).items()}
    second = service.run_github_evidence_preparation(confirmed=True, **inputs)
    after_unchanged = {key: path.read_bytes() for key, path in paths(tmp_path).items()}
    stale = service.get_github_evidence_preparation_status(
        **ready_inputs(tmp_path, body="changed saved evidence"),
    )
    changed = service.run_github_evidence_preparation(
        confirmed=True, **ready_inputs(tmp_path, body="changed saved evidence"),
    )
    assert first["status"] == "created" and first["preparation_complete"] is True
    assert second["status"] == "unchanged"
    assert before == after_unchanged
    assert stale["status"] == "ready_to_prepare" and stale["evidence_prepared"] is False
    assert changed["status"] == "updated"
    current = service.get_github_evidence_preparation_status(**ready_inputs(tmp_path, body="changed saved evidence"))
    assert current["status"] == "prepared" and current["evidence_prepared"] is True
    assert current["ready_for_retrieval_setup"] is True
    assert_product_safe(first); assert_product_safe(current)


def test_explicit_confirmation_is_required_and_failed_preflight_writes_nothing(tmp_path):
    inputs = ready_inputs(tmp_path)
    for confirmed in (False, None, "true", 1):
        result = service.run_github_evidence_preparation(confirmed=confirmed, **inputs)
        assert result["status"] == "blocked"
    assert not any(path.exists() for path in paths(tmp_path).values())


def test_partial_and_materializer_error_translate_without_internal_details(tmp_path):
    inputs = ready_inputs(tmp_path)
    partial = service.run_github_evidence_preparation(
        confirmed=True, **inputs,
        materializer=lambda **_: {"status": "partial", "raw_text": "secret"},
        readiness_inspector=lambda **_: {
            "status": "partial", "chunk_mapping_ready": False,
            "records_unresolved": 0, "projects_with_chunks": ["A"],
        },
    )
    failed = service.run_github_evidence_preparation(
        confirmed=True, **inputs,
        materializer=lambda **_: {"status": "error", "errors": ["C:\\private", "diff --git"]},
    )
    assert partial["status"] == "partial" and partial["preparation_complete"] is False
    assert failed["status"] == "error"
    assert_product_safe(partial); assert_product_safe(failed)


def test_readiness_failure_after_durable_materialization_is_degraded(tmp_path):
    inputs = ready_inputs(tmp_path)
    result = service.run_github_evidence_preparation(
        confirmed=True, **inputs, readiness_inspector=lambda **_: (_ for _ in ()).throw(RuntimeError("private")),
    )
    assert result["status"] == "degraded" and result["evidence_prepared"] is True
    assert all(path.exists() for path in paths(tmp_path).values())
    assert "private" not in json.dumps(result)


def test_concurrent_runs_execute_materializer_once_and_return_busy(tmp_path):
    inputs = ready_inputs(tmp_path)
    entered = threading.Event(); release = threading.Event(); calls = []
    def slow_materializer(**_kwargs):
        calls.append(1); entered.set(); release.wait(timeout=5); return {"status": "created"}
    results = []
    thread = threading.Thread(target=lambda: results.append(service.run_github_evidence_preparation(
        confirmed=True, **inputs, materializer=slow_materializer,
        readiness_inspector=lambda **_: {"status": "ready", "chunk_mapping_ready": True, "records_unresolved": 0, "projects_with_chunks": ["A"]},
    )))
    thread.start(); assert entered.wait(timeout=3)
    second = service.run_github_evidence_preparation(confirmed=True, **inputs, materializer=slow_materializer)
    release.set(); thread.join(timeout=5)
    assert second["status"] == "busy" and len(calls) == 1
    assert not thread.is_alive()


def test_materializer_accepts_validated_mapping_without_project_memory_repository(tmp_path):
    memory = project_memory("A")
    mapping = identity.authority_to_repository_mapping(authority(memory, link()))
    result = materializer.materialize_saved_github_evidence(
        saved_context=saved_context(), project_memory=memory, authoritative_mapping=mapping,
        feature_enabled=True, **paths(tmp_path),
    )
    assert result["status"] == "created" and result["project_ids"] == ["A"]
