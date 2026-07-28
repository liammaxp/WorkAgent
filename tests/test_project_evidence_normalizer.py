import importlib
import json
from pathlib import Path
import sys

import pytest

from backend.project_evidence_normalizer import (
    ProjectEvidenceIntegrityError,
    dedupe_project_evidence_inputs,
    normalize_and_dedupe_project_evidence_inputs,
    normalize_project_evidence_input,
    normalize_project_evidence_inputs,
)
from backend.project_evidence_input import load_project_evidence_inputs
from backend.project_evidence_models import ProjectEvidenceInput, EvidenceSourceRef


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def source(source_id="source-1", **changes):
    values = dict(
        source_type="github_evidence_chunk",
        source_id=source_id,
        project_id="workagent",
        content_hash=DIGEST_A,
        repo="owner/WorkAgent",
        file_path=r"backend\normalizer.py",
        symbol="normalize_input",
        commit_sha="abc123",
    )
    values.update(changes)
    return EvidenceSourceRef(**values)


def evidence(**changes):
    values = dict(
        project_id="workagent",
        input_type="github_evidence_card",
        title="Evidence normalization",
        summary="Normalizes structured evidence.",
        problem_signal="Duplicate structured values.",
        mechanism_signals=["retrieve", "validate"],
        implementation_signals=["read input", "build model"],
        impact_signals=["Supports deterministic evidence."],
        technical_tags=["Python", "validation"],
        source_refs=[source()],
        content_hash=DIGEST_B,
    )
    values.update(changes)
    return ProjectEvidenceInput(**values)


def test_bounded_text_normalization_preserves_punctuation_and_case():
    item = evidence()
    object.__setattr__(item, "title", "  Parse   HTTPResponse.validate()!  ")
    object.__setattr__(item, "summary", "Line one.\r\n\tLine   two?")
    normalized = normalize_project_evidence_input(item)
    assert normalized.title == "Parse HTTPResponse.validate()!"
    assert normalized.summary == "Line one. Line two?"
    assert "HTTPResponse.validate()" in normalized.title


def test_project_id_and_input_type_preserve_canonical_case():
    item = evidence()
    object.__setattr__(item, "project_id", " WorkAgent ")
    object.__setattr__(item, "input_type", " github_evidence_card ")
    object.__setattr__(item.source_refs[0], "project_id", "WorkAgent")
    normalized = normalize_project_evidence_input(item)
    assert normalized.project_id == "WorkAgent"
    assert normalized.input_type == "github_evidence_card"


def test_file_path_uses_existing_model_normalization():
    normalized = normalize_project_evidence_input(evidence(source_refs=[source(file_path=r"backend\project_evidence\item.py")]))
    assert normalized.source_refs[0].file_path == "backend/project_evidence/item.py"


@pytest.mark.parametrize(
    "field",
    ["mechanism_signals", "implementation_signals", "impact_signals"],
)
def test_process_signal_duplicates_and_blanks_removed_preserving_first(field):
    item = evidence()
    values = getattr(item, field)
    original = list(values)
    values.extend(["   ", f"  {original[0]}  "])
    normalized = normalize_project_evidence_input(item)
    assert getattr(normalized, field) == original


def test_technical_tags_are_set_like():
    item = evidence()
    item.technical_tags.extend(["python", " retrieval ", "Retrieval", ""])
    normalized = normalize_project_evidence_input(item)
    assert normalized.technical_tags == ["Python", "retrieval", "validation"]


def test_source_ref_order_preserved_and_exact_duplicates_removed():
    first = source("first", content_hash=DIGEST_A)
    second = source("second", content_hash=DIGEST_B)
    item = evidence(source_refs=[second, first])
    item.source_refs.append(EvidenceSourceRef.from_dict(second.to_dict()))
    normalized = normalize_project_evidence_input(item)
    assert [ref.source_id for ref in normalized.source_refs] == ["second", "first"]


def test_source_refs_with_partial_identity_match_remain_separate():
    first = source("first")
    second = source("second")
    normalized = normalize_project_evidence_input(evidence(source_refs=[first, second]))
    assert len(normalized.source_refs) == 2


def test_input_object_is_not_mutated():
    item = evidence()
    item.mechanism_signals.extend(["", "retrieve"])
    before = item.to_json()
    normalize_project_evidence_input(item)
    assert item.to_json() == before


def test_normalization_is_idempotent_and_serially_stable():
    once = normalize_project_evidence_input(evidence())
    twice = normalize_project_evidence_input(once)
    assert once.to_json() == twice.to_json()


def test_equivalent_normalized_records_have_same_id():
    one = evidence()
    two = evidence()
    object.__setattr__(two, "title", " Evidence   normalization ")
    assert normalize_project_evidence_input(one).input_id == normalize_project_evidence_input(two).input_id


@pytest.mark.parametrize("field", ["mechanism_signals", "implementation_signals"])
def test_process_order_changes_identity(field):
    one = evidence()
    two = evidence()
    object.__setattr__(two, field, list(reversed(getattr(two, field))))
    assert normalize_project_evidence_input(one).input_id != normalize_project_evidence_input(two).input_id


def test_tag_order_does_not_change_identity():
    one = evidence(technical_tags=["Python", "validation"])
    two = evidence(technical_tags=["validation", "Python"])
    assert normalize_project_evidence_input(one).input_id == normalize_project_evidence_input(two).input_id


def test_source_ref_order_is_identity_bearing():
    refs = [source("one", content_hash=DIGEST_A), source("two", content_hash=DIGEST_B)]
    one = evidence(source_refs=refs)
    two = evidence(source_refs=list(reversed(refs)))
    assert normalize_project_evidence_input(one).input_id != normalize_project_evidence_input(two).input_id


def test_materially_different_safe_content_changes_id():
    one = evidence(summary="First structured summary.")
    two = evidence(summary="Second structured summary.")
    assert normalize_project_evidence_input(one).input_id != normalize_project_evidence_input(two).input_id


def test_timestamps_and_absolute_machine_paths_do_not_enter_id():
    one_ref = source(file_path="C:/machine-one/repo/a.py", metadata={"generated_at": "2026-01-01T00:00:00Z"})
    two_ref = source(file_path="D:/machine-two/repo/a.py", metadata={"generated_at": "2027-01-01T00:00:00Z"})
    one = normalize_project_evidence_input(evidence(source_refs=[one_ref]))
    two = normalize_project_evidence_input(evidence(source_refs=[two_ref]))
    assert one.input_id == two.input_id
    assert one.content_hash == two.content_hash


def test_content_hash_is_recomputed_after_normalization():
    item = evidence(content_hash=DIGEST_A)
    normalized = normalize_project_evidence_input(item)
    assert normalized.content_hash != DIGEST_A
    changed = normalize_project_evidence_input(evidence(summary="Changed."))
    assert normalized.content_hash != changed.content_hash


def test_normalized_collection_order_is_deterministic():
    alpha = evidence(project_id="alpha", source_refs=[source(project_id="alpha")])
    zeta = evidence(project_id="zeta", source_refs=[source(project_id="zeta")])
    assert [item.project_id for item in normalize_project_evidence_inputs([zeta, alpha])] == ["alpha", "zeta"]


def test_identical_records_collapse_as_exact_duplicate():
    item = evidence()
    output, report = dedupe_project_evidence_inputs([item, ProjectEvidenceInput.from_dict(item.to_dict())])
    assert len(output) == 1
    assert report.exact_duplicates_removed == 1
    assert report.repeated_source_records_removed == 0


def test_repeated_traversal_with_stale_id_collapses():
    one = evidence()
    two = evidence(input_id="pei_stale_traversal_copy")
    output, report = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 1
    assert report.repeated_source_records_removed == 1


def test_duplicates_across_projects_remain_separate():
    one = evidence(project_id="workagent", source_refs=[source(project_id="workagent")])
    two = evidence(project_id="WorkAgent", source_refs=[source(project_id="WorkAgent")])
    output, _ = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 2


def test_similar_summary_from_different_source_types_remains_separate():
    one = evidence(input_type="project_memory", source_refs=[source(source_type="project_memory")])
    two = evidence(input_type="github_evidence_card")
    output, report = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 2
    assert report.retained_cross_source_records == 1


@pytest.mark.parametrize(
    "change",
    [
        {"commit_sha": "different"},
        {"file_path": "backend/different.py"},
        {"symbol": "DifferentSymbol"},
    ],
)
def test_distinct_provenance_remains_separate(change):
    one = evidence(source_refs=[source()])
    two = evidence(source_refs=[source(**change)])
    output, _ = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 2


def test_same_source_identity_with_different_content_remains_separate():
    one = evidence(summary="First version.", content_hash=DIGEST_A)
    two = evidence(summary="Second version.", content_hash=DIGEST_B)
    output, _ = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 2


def test_card_and_capability_and_project_memory_remain_separate():
    card = evidence(input_type="github_evidence_card")
    capability = evidence(input_type="github_evidence_capability_fact", source_refs=[source(source_type="github_evidence_capability_fact")])
    memory = evidence(input_type="project_memory", source_refs=[source(source_type="project_memory")])
    output, _ = dedupe_project_evidence_inputs([card, capability, memory])
    assert len(output) == 3


def test_similar_wording_is_not_semantically_merged():
    one = evidence(summary="added validation before export", content_hash=DIGEST_A)
    two = evidence(summary="validated generated output before LaTeX export", content_hash=DIGEST_B)
    output, _ = dedupe_project_evidence_inputs([one, two])
    assert len(output) == 2


def test_deterministic_winner_and_output_do_not_depend_on_input_order():
    original = evidence()
    stale = evidence(input_id="pei_stale_copy")
    alpha = evidence(project_id="alpha", source_refs=[source(project_id="alpha")])
    forward, forward_report = dedupe_project_evidence_inputs([stale, alpha, original])
    reverse, reverse_report = dedupe_project_evidence_inputs([original, alpha, stale])
    assert [item.to_json() for item in forward] == [item.to_json() for item in reverse]
    assert forward_report.to_dict() == reverse_report.to_dict()


def test_report_counters_include_in_record_duplicates():
    item = evidence()
    item.mechanism_signals.extend(["retrieve", ""])
    item.implementation_signals.append("read input")
    item.impact_signals.append("Supports deterministic evidence.")
    item.technical_tags.append("python")
    item.source_refs.append(EvidenceSourceRef.from_dict(item.source_refs[0].to_dict()))
    output, report = normalize_and_dedupe_project_evidence_inputs([item, item])
    assert len(output) == 1
    assert report.input_count == 2
    assert report.normalized_count == 2
    assert report.output_count == 1
    assert report.exact_duplicates_removed == 1
    assert report.duplicate_source_refs_removed == 2
    assert report.duplicate_signal_values_removed == 10


def test_decisions_are_bounded_and_contain_no_evidence_text():
    item = evidence(summary="RAW_CONTENT_SENTINEL")
    _, report = dedupe_project_evidence_inputs([item, item])
    serialized = json.dumps(report.to_dict())
    assert len(report.decisions) == 1
    assert "RAW_CONTENT_SENTINEL" not in serialized


def test_same_declared_input_id_with_different_payload_is_detected_safely():
    one = evidence(summary="First payload RAW_SENTINEL")
    two = evidence(summary="Second payload SECRET_SENTINEL")
    object.__setattr__(two, "input_id", one.input_id)
    with pytest.raises(ProjectEvidenceIntegrityError, match="same_input_id_different_payload") as captured:
        dedupe_project_evidence_inputs([one, two])
    assert "RAW_SENTINEL" not in str(captured.value)
    assert "SECRET_SENTINEL" not in str(captured.value)


def test_same_declared_hash_same_source_with_different_payload_is_detected():
    one = evidence(summary="First payload.")
    two = evidence(summary="Second payload.")
    object.__setattr__(two, "content_hash", one.content_hash)
    with pytest.raises(ProjectEvidenceIntegrityError, match="same_content_hash_different_payload"):
        dedupe_project_evidence_inputs([one, two])


def test_forbidden_raw_and_secret_fields_remain_rejected():
    payload = source().to_dict()
    payload["raw_patch"] = "patch"
    with pytest.raises(ValueError, match="unknown"):
        EvidenceSourceRef.from_dict(payload)
    with pytest.raises(ValueError, match="forbidden"):
        source(metadata={"nested": {"api_key": "secret"}})


def test_oversized_content_is_not_truncated():
    item = evidence()
    object.__setattr__(item, "summary", "x" * 2001)
    with pytest.raises(ValueError, match="maximum length"):
        normalize_project_evidence_input(item)


def test_normalization_performs_no_file_writes(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text("{}", encoding="utf-8")
    before = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    dedupe_project_evidence_inputs([evidence()])
    after = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert before == after


def test_import_has_no_runtime_or_external_side_effects():
    module_name = "backend.project_evidence_normalizer"
    sys.modules.pop(module_name, None)
    before = set(sys.modules)
    importlib.import_module(module_name)
    newly_loaded = set(sys.modules) - before
    assert "backend.api_server" not in newly_loaded
    assert "backend.project_change_pipeline" not in newly_loaded
    assert not any(name.startswith(("chromadb", "openai", "sqlite3", "requests")) for name in newly_loaded)


def test_real_step2_output_passes_read_only_through_step3():
    inputs, _ = load_project_evidence_inputs()
    output, report = dedupe_project_evidence_inputs(inputs)
    assert report.input_count == 449
    assert report.output_count == 449
    assert len(output) == 449
