import importlib
import json
from pathlib import Path
import sys

import pytest

from backend.project_evidence_input import (
    ProjectEvidenceInputSourcePaths,
    load_project_evidence_inputs,
)


DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


def disabled_paths(**changes) -> ProjectEvidenceInputSourcePaths:
    values = dict(github_evidence_memory_dir=None, project_change_memory_path=None, project_memory_path=None, compact_facts_path=None)
    values.update(changes)
    return ProjectEvidenceInputSourcePaths(**values)


def chunk(chunk_id="chunk-1", project_id="workagent", digest=DIGEST, **changes):
    value = dict(chunk_id=chunk_id, source_id="raw-1", project_id=project_id, repo="owner/WorkAgent", path=r"backend\adapter.py", symbol="load", chunk_type="function", start_line=10, end_line=20, hash=digest, summary="Safe structured summary", technical_tags=["Python"])
    value.update(changes)
    return value


def github_evidence_card(evidence_id="evidence-1", project_id="workagent", source_ids=None, **changes):
    value = dict(evidence_id=evidence_id, project_id=project_id, source_chunk_ids=source_ids or ["chunk-1"], problem="Inputs used unsafe source text.", mechanism="Added a bounded adapter.", implementation_details=["read", "validate", "convert"], safe_impact="Supports bounded local inputs.", resume_angle="Bounded evidence adapter", allowed_claims=["Implemented a local adapter."], forbidden_claims=[], metadata={"technical_tags": ["Python", "python", "JSON"]})
    value.update(changes)
    return value


def make_github_evidence_dir(path: Path, cards=None, chunks=None, summaries=None, capabilities=None) -> Path:
    write_jsonl(path / "evidence_chunks.jsonl", chunks if chunks is not None else [chunk()])
    write_jsonl(path / "evidence_cards.jsonl", cards if cards is not None else [github_evidence_card()])
    write_jsonl(path / "raw_change_summaries.jsonl", summaries if summaries is not None else [])
    write_jsonl(path / "capability_facts.jsonl", capabilities if capabilities is not None else [])
    return path


def project_change_payload(records=None, cards=None, capabilities=None):
    return {
        "schema_version": "project_change_memory.v1",
        "updated_at": None,
        "projects": {
            "workagent": {
                "project_id": "workagent",
                "raw_change_summaries": records if records is not None else [{"change_id": "change-1", "project_id": "workagent", "repo": "owner/WorkAgent", "commit_sha": "abc123", "file_path": "backend/a.py", "symbols_changed": ["retrieve", "validate"], "raw_change_types": ["validation_logic_update"], "what_changed": "Added structured validation."}],
                "evidence_cards": cards if cards is not None else [],
                "capability_facts": capabilities if capabilities is not None else [],
            }
        },
    }


def project_memory(projects=None):
    return {
        "version": 1,
        "purpose": "structured project memory",
        "source": "repository-analysis",
        "projects": projects if projects is not None else [{
            "project_id": "workagent",
            "project_name": "WorkAgent",
            "identity": {"core_problem": "Evidence inputs could be unsafe.", "core_value": "Converts structured evidence safely."},
            "tech_stack": ["Python", "JSON", "python"],
            "key_modules": ["adapter", "models"],
            "workflows": ["read", "validate", "convert"],
            "confirmed_features": ["bounded inputs"],
            "resume_relevant_claims": ["Implemented bounded evidence adapters."],
        }],
    }


def compact_payload(**record_changes):
    record = {
        "id": "compact-1",
        "project_name": "WorkAgent",
        "repo_name": "owner/WorkAgent",
        "source_hash": OTHER_DIGEST,
        "compact_facts_json": {
            "projectName": "WorkAgent",
            "projectSummary": "Compact structured project facts.",
            "technicalStack": ["JSON", "Python", "python"],
            "keyModules": ["reader", "validator"],
            "resumeRelevantClaims": ["Implemented compact fact loading."],
            "riskFlags": ["Do not infer metrics."],
        },
    }
    record.update(record_changes)
    return {"entry-hash": record}


def serialized(result):
    inputs, warnings = result
    return ([item.to_json() for item in inputs], [item.to_json() for item in warnings])


def test_loads_valid_github_evidence_artifact(tmp_path):
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence")
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert not warnings
    assert len(inputs) == 1
    assert inputs[0].input_type == "github_evidence_card"
    assert inputs[0].source_refs[0].source_id == "chunk-1"
    assert inputs[0].source_refs[0].file_path == "backend/adapter.py"


def test_github_evidence_legacy_zero_line_sentinels_become_absent_provenance(tmp_path):
    memory_dir = make_github_evidence_dir(
        tmp_path / "github_evidence",
        chunks=[chunk(start_line=0, end_line=0)],
    )
    inputs, warnings = load_project_evidence_inputs(
        source_paths=disabled_paths(github_evidence_memory_dir=memory_dir)
    )
    assert not warnings
    assert inputs[0].source_refs[0].start_line is None
    assert inputs[0].source_refs[0].end_line is None


def test_loads_valid_project_change_artifact(tmp_path):
    path = tmp_path / "project_change.json"
    write_json(path, project_change_payload())
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_change_memory_path=path))
    assert not warnings
    assert [item.input_type for item in inputs] == ["project_change_raw_change_summary"]
    assert inputs[0].source_refs[0].commit_sha == "abc123"


def test_loads_valid_project_memory(tmp_path):
    path = tmp_path / "project_memory.json"
    write_json(path, project_memory())
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert not warnings
    assert inputs[0].problem_signal == "Evidence inputs could be unsafe."
    assert inputs[0].mechanism_signals == ["read", "validate", "convert"]


def test_loads_compact_facts_with_established_project_mapping(tmp_path):
    project_path = tmp_path / "project_memory.json"; compact_path = tmp_path / "compact.json"
    write_json(project_path, project_memory()); write_json(compact_path, compact_payload())
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=project_path, compact_facts_path=compact_path))
    assert not warnings
    compact = next(item for item in inputs if item.input_type == "project_compact_facts")
    assert compact.project_id == "workagent"
    assert compact.implementation_signals == ["reader", "validator"]


def test_filters_by_normalized_project_id(tmp_path):
    records = project_memory()["projects"] + [{"project_id": "other", "project_name": "Other", "identity": {"core_value": "Other summary"}}]
    path = tmp_path / "project.json"; write_json(path, project_memory(records))
    inputs, _ = load_project_evidence_inputs("  workagent  ", source_paths=disabled_paths(project_memory_path=path))
    assert {item.project_id for item in inputs} == {"workagent"}


def test_missing_optional_source_file_warns_safely(tmp_path):
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_change_memory_path=tmp_path / "missing.json"))
    assert not inputs
    assert [item.code for item in warnings] == ["source_file_missing"]
    assert str(tmp_path) not in warnings[0].message


def test_invalid_json_does_not_raise_or_echo_content(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"secret":"DO_NOT_ECHO"', encoding="utf-8")
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert not inputs
    assert warnings[0].code == "source_json_invalid"
    assert "DO_NOT_ECHO" not in warnings[0].to_json()


@pytest.mark.parametrize("value", ["wrong", ["wrong"]])
def test_invalid_root_type_warns(tmp_path, value):
    path = tmp_path / "source.json"; write_json(path, value)
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert not inputs
    assert warnings[0].code == "source_schema_invalid"


def test_malformed_record_does_not_block_valid_record(tmp_path):
    path = tmp_path / "project.json"; payload = project_memory(); payload["projects"].insert(0, "not-an-object"); write_json(path, payload)
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert len(inputs) == 1
    assert any(item.code == "record_not_object" for item in warnings)


def test_missing_project_id_skips_record(tmp_path):
    path = tmp_path / "project.json"; write_json(path, project_memory([{"project_name": "No ID", "identity": {"core_value": "summary"}}]))
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert not inputs
    assert warnings[0].code == "missing_project_id"


def test_oversized_malformed_identifier_does_not_escape_warning_bounds(tmp_path):
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", cards=[github_evidence_card(evidence_id="x" * 1000, project_id="")])
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert not inputs
    assert warnings[0].code == "missing_project_id"
    assert warnings[0].source_id is None


def test_project_id_mismatch_in_github_evidence_provenance(tmp_path):
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", chunks=[chunk(project_id="other")])
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert not inputs
    assert any(item.code == "project_id_mismatch" for item in warnings)


@pytest.mark.parametrize("chunks,expected", [([], "source_ref_invalid"), ([chunk(digest="bad")], "missing_content_hash")])
def test_missing_or_invalid_source_provenance(tmp_path, chunks, expected):
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", chunks=chunks)
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert not inputs
    assert any(item.code == expected for item in warnings)


def test_deterministic_ids_and_repeated_serialization(tmp_path):
    path = tmp_path / "project.json"; write_json(path, project_memory())
    paths = disabled_paths(project_memory_path=path)
    first = load_project_evidence_inputs(source_paths=paths); second = load_project_evidence_inputs(source_paths=paths)
    assert first[0][0].input_id == second[0][0].input_id
    assert serialized(first) == serialized(second)


def test_materially_different_content_changes_input_id(tmp_path):
    path = tmp_path / "project.json"; write_json(path, project_memory())
    first, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    changed = project_memory(); changed["projects"][0]["identity"]["core_value"] = "Materially changed summary."
    write_json(path, changed)
    second, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert first[0].input_id != second[0].input_id


def test_deterministic_output_order(tmp_path):
    projects = [
        {"project_id": "zeta", "project_name": "Zeta", "identity": {"core_value": "Zeta summary"}},
        {"project_id": "alpha", "project_name": "Alpha", "identity": {"core_value": "Alpha summary"}},
    ]
    path = tmp_path / "project.json"; write_json(path, project_memory(projects))
    inputs, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    assert [item.project_id for item in inputs] == ["alpha", "zeta"]


def test_raw_patch_and_secrets_are_not_copied(tmp_path):
    card = github_evidence_card(raw_patch="COMPLETE_PATCH_SENTINEL", metadata={"api_key": "SECRET_SENTINEL", "technical_tags": ["Python"]})
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", cards=[card], chunks=[chunk(raw_text="RAW_TEXT_SENTINEL")])
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    output = json.dumps([item.to_dict() for item in inputs] + [item.to_dict() for item in warnings])
    assert "COMPLETE_PATCH_SENTINEL" not in output
    assert "SECRET_SENTINEL" not in output
    assert "RAW_TEXT_SENTINEL" not in output


def test_oversized_content_skips_only_bad_record(tmp_path):
    cards = [github_evidence_card("bad", mechanism="x" * 1001), github_evidence_card("good")]
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", cards=cards)
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert [item.source_refs[0].source_id for item in inputs] == ["chunk-1"]
    assert len(inputs) == 1
    assert any(item.code == "record_validation_failed" and item.source_id == "bad" for item in warnings)


def test_source_refs_retain_logical_order(tmp_path):
    chunks = [chunk("chunk-1", digest=DIGEST), chunk("chunk-2", digest=OTHER_DIGEST)]
    memory_dir = make_github_evidence_dir(tmp_path / "github_evidence", cards=[github_evidence_card(source_ids=["chunk-2", "chunk-1"])], chunks=chunks)
    inputs, _ = load_project_evidence_inputs(source_paths=disabled_paths(github_evidence_memory_dir=memory_dir))
    assert [ref.source_id for ref in inputs[0].source_refs] == ["chunk-2", "chunk-1"]


def test_tags_are_normalized_deduplicated_and_order_insensitive(tmp_path):
    first_path = tmp_path / "first.json"; second_path = tmp_path / "second.json"
    first = project_memory(); second = project_memory()
    first["projects"][0]["tech_stack"] = ["validation", "retrieval", "Validation"]
    second["projects"][0]["tech_stack"] = ["retrieval", "validation"]
    write_json(first_path, first); write_json(second_path, second)
    one, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=first_path)); two, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=second_path))
    assert one[0].technical_tags == ["retrieval", "validation"]
    assert one[0].input_id == two[0].input_id


def test_implementation_signals_preserve_order_and_affect_id(tmp_path):
    first_path = tmp_path / "first.json"; second_path = tmp_path / "second.json"
    first = project_memory(); second = project_memory()
    first["projects"][0]["confirmed_features"] = ["retrieve", "validate"]
    second["projects"][0]["confirmed_features"] = ["validate", "retrieve"]
    write_json(first_path, first); write_json(second_path, second)
    one, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=first_path)); two, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=second_path))
    assert one[0].implementation_signals[:2] == ["retrieve", "validate"]
    assert one[0].input_id != two[0].input_id


def test_no_cross_source_semantic_deduplication(tmp_path):
    project_path = tmp_path / "project.json"; compact_path = tmp_path / "compact.json"
    write_json(project_path, project_memory()); write_json(compact_path, compact_payload())
    inputs, _ = load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=project_path, compact_facts_path=compact_path))
    assert len(inputs) == 2
    assert {item.input_type for item in inputs} == {"project_memory", "project_compact_facts"}


def test_bad_source_does_not_block_another_source(tmp_path):
    bad = tmp_path / "bad.json"; good = tmp_path / "good.json"
    bad.write_text("not-json", encoding="utf-8"); write_json(good, project_memory())
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_change_memory_path=bad, project_memory_path=good))
    assert len(inputs) == 1
    assert any(item.code == "source_json_invalid" for item in warnings)


def test_call_performs_no_file_writes(tmp_path):
    path = tmp_path / "project.json"; write_json(path, project_memory())
    before = {item.relative_to(tmp_path): item.stat().st_mtime_ns for item in tmp_path.rglob("*") if item.is_file()}
    load_project_evidence_inputs(source_paths=disabled_paths(project_memory_path=path))
    after = {item.relative_to(tmp_path): item.stat().st_mtime_ns for item in tmp_path.rglob("*") if item.is_file()}
    assert before == after


def test_import_has_no_runtime_or_io_side_effects():
    module_name = "backend.project_evidence_input"
    sys.modules.pop(module_name, None)
    before = set(sys.modules)
    importlib.import_module(module_name)
    newly_loaded = set(sys.modules) - before
    assert "backend.api_server" not in newly_loaded
    assert "backend.evidence_memory" not in newly_loaded
    assert "backend.project_change_pipeline" not in newly_loaded
    assert not any(name.startswith(("chromadb", "openai", "requests")) for name in newly_loaded)


def test_unsupported_schema_version_is_not_interpreted(tmp_path):
    path = tmp_path / "project_change.json"; payload = project_change_payload(); payload["schema_version"] = "project_change_memory.v99"; write_json(path, payload)
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(project_change_memory_path=path))
    assert not inputs
    assert warnings[0].code == "unsupported_source_schema"


def test_compact_fact_without_established_mapping_is_skipped(tmp_path):
    path = tmp_path / "compact.json"; write_json(path, compact_payload())
    inputs, warnings = load_project_evidence_inputs(source_paths=disabled_paths(compact_facts_path=path))
    assert not inputs
    assert warnings[0].code == "missing_project_id"
