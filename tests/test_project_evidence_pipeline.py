from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import backend.project_evidence_pipeline as pipeline
from backend.project_capability_taxonomy import ProjectCapabilityTaxonomyValidationReport
from backend.project_evidence_input import ProjectEvidenceInputSourcePaths
from backend.project_evidence_models import ProjectEvidenceInput, ProjectEvidencePipelineWarning, EvidenceSourceRef
from backend.project_evidence_memory import load_project_evidence_memory


ENABLED = {pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: "true"}
DISABLED_PATHS = ProjectEvidenceInputSourcePaths(
    github_evidence_memory_dir=None,
    project_change_memory_path=None,
    project_memory_path=None,
    compact_facts_path=None,
)


def evidence_input(project_id: str = "alpha", **changes) -> ProjectEvidenceInput:
    reference = EvidenceSourceRef(
        source_type="project_change_evidence_card",
        source_id=f"source-{project_id}",
        project_id=project_id,
        content_hash="a" * 64,
        file_path="backend/memory.py",
        symbol="persist_memory",
    )
    values = {
        "project_id": project_id,
        "input_type": "project_change_evidence_card",
        "title": "Deterministic project memory persistence",
        "summary": "Structured evidence for a bounded persistence workflow.",
        "problem_signal": "Project memory needed validated replacement.",
        "mechanism_signals": ["atomic persistence", "schema validation"],
        "implementation_signals": [
            "backend/memory.py::persist_memory",
            "write temporary file then validate and replace",
        ],
        "impact_signals": ["Keeps structured replacement atomic."],
        "technical_tags": ["Python", "persistence"],
        "source_refs": [reference],
        "content_hash": "b" * 64,
    }
    values.update(changes)
    return ProjectEvidenceInput(**values)


def install_loader(monkeypatch, items=None, warnings=()):
    records = list([evidence_input()] if items is None else items)
    warning_records = list(warnings)

    def load(*_args, **_kwargs):
        return list(records), list(warning_records)

    monkeypatch.setattr(pipeline.input_adapter, "load_project_evidence_inputs", load)
    return DISABLED_PATHS


def run_fixture(monkeypatch, tmp_path, *, output_name="project_evidence.json", persist=True):
    paths = install_loader(monkeypatch)
    return pipeline.run_project_evidence_pipeline(
        source_paths=paths,
        output_path=tmp_path / output_name,
        persist=persist,
        environ=ENABLED,
    )


@pytest.mark.parametrize("value", [None, "", "0", "false", "FALSE", "no", "off", " Off "])
def test_disabled_feature_flag_values(value):
    environ = {} if value is None else {pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: value}
    assert not pipeline.is_project_evidence_memory_enabled(environ)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_enabled_feature_flag_values(value):
    assert pipeline.is_project_evidence_memory_enabled({pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: value})


def test_invalid_flag_is_safely_disabled_with_stable_warning():
    result = pipeline.run_project_evidence_pipeline(
        environ={pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: "sometimes"}
    )
    assert result.status == "disabled"
    assert [warning.code for warning in result.warnings] == ["invalid_feature_flag"]


def test_disabled_pipeline_performs_no_reads_or_writes_and_preserves_file(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    path.write_bytes(b"existing")
    before = path.read_bytes()
    monkeypatch.setattr(
        pipeline.input_adapter, "load_project_evidence_inputs",
        lambda *_a, **_k: pytest.fail("source read attempted"),
    )
    monkeypatch.setattr(
        pipeline.project_memory, "load_project_evidence_memory",
        lambda *_a, **_k: pytest.fail("artifact read attempted"),
    )
    result = pipeline.run_project_evidence_pipeline(
        output_path=path, environ={pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: "0"}
    )
    assert result.status == "disabled"
    assert all(stage.status == "skipped" for stage in result.stages)
    assert path.read_bytes() == before


def test_stage_order_is_complete_and_deterministic(tmp_path, monkeypatch):
    result = run_fixture(monkeypatch, tmp_path, persist=False)
    assert tuple(stage.stage for stage in result.stages) == pipeline.PROJECT_EVIDENCE_PIPELINE_STAGE_ORDER
    assert len(result.stages) == len(set(stage.stage for stage in result.stages))


def test_stage_outputs_feed_the_next_accepted_interfaces(tmp_path, monkeypatch):
    original_dedupe = pipeline.evidence_normalizer.dedupe_project_evidence_inputs
    original_synthesize = pipeline.evidence_synthesizer.synthesize_project_evidence_facts
    original_score = pipeline.evidence_scoring.score_project_evidence_facts
    original_signals = pipeline.capability_extractor.extract_project_evidence_fact_signals_many
    original_capabilities = pipeline.capability_extractor.extract_project_evidence_capabilities_by_project
    seen = {}
    supplied = [evidence_input()]

    monkeypatch.setattr(
        pipeline.input_adapter, "load_project_evidence_inputs", lambda **_kwargs: (supplied, [])
    )

    def dedupe(items):
        seen["dedupe_input"] = items
        output = original_dedupe(items)
        seen["normalized"] = output[0]
        return output

    def synthesize(items):
        assert items is seen["normalized"]
        output = original_synthesize(items)
        seen["facts"] = output[0]
        return output

    def score(items):
        assert items is seen["facts"]
        output = original_score(items)
        seen["scored"] = output[0]
        return output

    def signals(items):
        if items is seen["scored"]:
            seen["direct_signal_input"] = items
        return original_signals(items)

    def capabilities(items):
        assert items is seen["scored"]
        return original_capabilities(items)

    monkeypatch.setattr(pipeline.evidence_normalizer, "dedupe_project_evidence_inputs", dedupe)
    monkeypatch.setattr(pipeline.evidence_synthesizer, "synthesize_project_evidence_facts", synthesize)
    monkeypatch.setattr(pipeline.evidence_scoring, "score_project_evidence_facts", score)
    monkeypatch.setattr(pipeline.capability_extractor, "extract_project_evidence_fact_signals_many", signals)
    monkeypatch.setattr(pipeline.capability_extractor, "extract_project_evidence_capabilities_by_project", capabilities)
    result = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS,
        output_path=tmp_path / "memory.json",
        persist=False,
        environ=ENABLED,
    )
    assert result.status == "degraded"
    assert seen["dedupe_input"] is supplied
    assert seen["direct_signal_input"] is seen["scored"]


def test_stage_counts_propagate_without_content(tmp_path, monkeypatch):
    result = run_fixture(monkeypatch, tmp_path, persist=False)
    assert result.input_count == result.normalized_input_count == 1
    assert result.evidence_fact_count == 1
    assert result.capability_fact_count == 0
    assert result.claim_boundary_count >= 1
    assert result.project_count == 1
    assert all(
        isinstance(value, (bool, int, str))
        for stage in result.stages for value in stage.details.values()
    )


def test_enabled_empty_returns_empty_without_creating_artifact(tmp_path, monkeypatch):
    install_loader(monkeypatch, items=[])
    path = tmp_path / "memory.json"
    result = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
    )
    assert result.status == "empty"
    assert not path.exists()
    assert result.stages[0].status == "empty"
    assert all(stage.status == "skipped" for stage in result.stages[1:])


def test_empty_run_preserves_previous_valid_artifact(tmp_path, monkeypatch):
    first = run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    before = path.read_bytes()
    install_loader(monkeypatch, items=[])
    result = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
    )
    assert first.persisted
    assert result.status == "empty"
    assert result.previous_artifact_preserved
    assert path.read_bytes() == before


def test_missing_optional_project_change_source_warns_without_error(tmp_path):
    paths = ProjectEvidenceInputSourcePaths(
        github_evidence_memory_dir=None,
        project_change_memory_path=tmp_path / "missing-project_change.json",
        project_memory_path=None,
        compact_facts_path=None,
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths,
        output_path=tmp_path / "out.json",
        environ=ENABLED,
    )
    assert result.status == "empty"
    assert "optional_project_change_artifact_missing" in {warning.code for warning in result.warnings}
    assert not result.errors


def test_valid_pipeline_creates_then_returns_unchanged(tmp_path, monkeypatch):
    first = run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    first_bytes = path.read_bytes()
    second = run_fixture(monkeypatch, tmp_path)
    assert first.status in {"ready", "degraded"}
    assert first.persistence_status == "created"
    assert second.status == "unchanged"
    assert second.persistence_status == "unchanged"
    assert first.content_hash == second.content_hash
    assert path.read_bytes() == first_bytes
    assert load_project_evidence_memory(path).status == "ready"


def test_zero_capabilities_and_metric_evidence_are_nonfatal(tmp_path, monkeypatch):
    result = run_fixture(monkeypatch, tmp_path)
    codes = {warning.code for warning in result.warnings}
    assert result.capability_fact_count == 0
    assert {"capability_facts_empty", "metric_evidence_empty"} <= codes
    assert result.status in {"degraded", "unchanged"}
    assert not result.errors


@pytest.mark.parametrize(
    ("stage", "module", "attribute"),
    [
        ("load_inputs", "input_adapter", "load_project_evidence_inputs"),
        ("normalize_dedupe", "evidence_normalizer", "dedupe_project_evidence_inputs"),
        ("synthesize_evidence", "evidence_synthesizer", "synthesize_project_evidence_facts"),
        ("score_evidence", "evidence_scoring", "score_project_evidence_facts"),
        ("extract_capabilities", "capability_extractor", "extract_project_evidence_capabilities_by_project"),
    ],
)
def test_unexpected_stage_failure_stops_later_stages_with_safe_error(
    stage, module, attribute, tmp_path, monkeypatch
):
    paths = install_loader(monkeypatch)
    target = getattr(pipeline, module)
    monkeypatch.setattr(
        target, attribute,
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("SECRET traceback C:/private")),
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "memory.json", environ=ENABLED
    )
    assert result.status == "error"
    assert result.errors == ("internal_pipeline_error",)
    failed = next(item for item in result.stages if item.stage == stage)
    assert failed.status == "error"
    assert all(
        item.status == "skipped"
        for item in result.stages[result.stages.index(failed) + 1:]
    )
    serialized = json.dumps(result.to_dict())
    assert "SECRET" not in serialized and "C:/private" not in serialized and "Traceback" not in serialized


def test_normalization_integrity_conflict_has_specific_safe_code(tmp_path, monkeypatch):
    paths = install_loader(monkeypatch)
    monkeypatch.setattr(
        pipeline.evidence_normalizer,
        "dedupe_project_evidence_inputs",
        lambda *_a, **_k: (_ for _ in ()).throw(
            pipeline.evidence_normalizer.ProjectEvidenceIntegrityError(
                "same_id_different_payload",
                project_id="alpha",
                input_type="project_change_evidence_card",
                input_id="pei_safe",
            )
        ),
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "memory.json", environ=ENABLED
    )
    assert result.errors == ("normalization_integrity_conflict",)


def test_taxonomy_failure_stops_extraction_and_persistence(tmp_path, monkeypatch):
    paths = install_loader(monkeypatch)
    monkeypatch.setattr(
        pipeline.capability_taxonomy,
        "validate_project_capability_taxonomy",
        lambda: ProjectCapabilityTaxonomyValidationReport(
            valid=False, capability_count=0, signal_count=0, alias_count=0,
            errors=("invalid_taxonomy",), warnings=(),
        ),
    )
    monkeypatch.setattr(
        pipeline.capability_extractor,
        "extract_project_evidence_fact_signals_many",
        lambda *_a, **_k: pytest.fail("extraction must not run"),
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "memory.json", environ=ENABLED
    )
    assert result.status == "error"
    assert result.errors == ("taxonomy_validation_failed",)
    assert next(stage for stage in result.stages if stage.stage == "extract_signals").status == "skipped"
    assert not (tmp_path / "memory.json").exists()


def test_boundary_validation_failure_stops_persistence(tmp_path, monkeypatch):
    paths = install_loader(monkeypatch)
    original = pipeline.claim_boundary.validate_project_claim_boundary
    calls = {"count": 0}

    def invalid_once(*args, **kwargs):
        calls["count"] += 1
        result = original(*args, **kwargs)
        return SimpleNamespace(valid=False, errors=("invalid",)) if calls["count"] == 1 else result

    monkeypatch.setattr(pipeline.claim_boundary, "validate_project_claim_boundary", invalid_once)
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "memory.json", environ=ENABLED
    )
    assert result.errors == ("boundary_validation_failed",)
    assert next(stage for stage in result.stages if stage.stage == "persist_project_memory").status == "skipped"


def test_persistence_failure_and_round_trip_failure_are_errors(tmp_path, monkeypatch):
    paths = install_loader(monkeypatch)
    monkeypatch.setattr(
        pipeline.project_memory,
        "persist_project_evidence_memory",
        lambda *_a, **_k: SimpleNamespace(status="failed"),
    )
    failed = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "failed.json", environ=ENABLED
    )
    assert failed.errors == ("persistence_failed",)

    monkeypatch.setattr(
        pipeline.project_memory,
        "persist_project_evidence_memory",
        lambda *_a, **_k: SimpleNamespace(
            status="created", bytes_written=10, previous_artifact_preserved=False,
            round_trip_validated=False,
        ),
    )
    round_trip = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "round-trip.json", environ=ENABLED
    )
    assert round_trip.errors == ("round_trip_validation_failed",)
    assert round_trip.status == "error"


def test_snapshot_validation_failure_stops_persistence(tmp_path, monkeypatch):
    paths = install_loader(monkeypatch)
    original_builder = pipeline.project_memory.build_project_evidence_memory_snapshot
    captured = {}

    def capture(*args, **kwargs):
        captured["snapshot"] = original_builder(*args, **kwargs)
        return captured["snapshot"]

    monkeypatch.setattr(pipeline.project_memory, "build_project_evidence_memory_snapshot", capture)
    baseline = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "baseline.json",
        persist=False, environ=ENABLED,
    )
    assert baseline.content_hash
    monkeypatch.setattr(
        pipeline.project_memory, "build_project_evidence_memory_snapshot",
        lambda *_a, **_k: captured["snapshot"],
    )
    monkeypatch.setattr(
        pipeline.project_memory, "validate_project_evidence_memory_snapshot",
        lambda *_a, **_k: SimpleNamespace(valid=False, errors=("invalid",)),
    )
    monkeypatch.setattr(
        pipeline.project_memory, "persist_project_evidence_memory",
        lambda *_a, **_k: pytest.fail("invalid snapshot must not persist"),
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=tmp_path / "invalid.json", environ=ENABLED
    )
    assert result.errors == ("snapshot_validation_failed",)
    assert next(stage for stage in result.stages if stage.stage == "persist_project_memory").status == "skipped"


def test_previous_valid_artifact_survives_pipeline_failure(tmp_path, monkeypatch):
    run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    before = path.read_bytes()
    paths = install_loader(monkeypatch)
    monkeypatch.setattr(
        pipeline.evidence_synthesizer,
        "synthesize_project_evidence_facts",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    result = pipeline.run_project_evidence_pipeline(
        source_paths=paths, output_path=path, environ=ENABLED
    )
    assert result.status == "error"
    assert result.previous_artifact_preserved
    assert result.next_recommended_action == "use_existing_artifact"
    assert path.read_bytes() == before


def test_previous_valid_artifact_survives_load_and_persistence_failures(tmp_path, monkeypatch):
    run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    before = path.read_bytes()
    monkeypatch.setattr(
        pipeline.input_adapter,
        "load_project_evidence_inputs",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("source failure")),
    )
    load_failed = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
    )
    assert load_failed.status == "error" and load_failed.previous_artifact_preserved
    assert path.read_bytes() == before

    install_loader(monkeypatch)
    monkeypatch.setattr(
        pipeline.project_memory,
        "persist_project_evidence_memory",
        lambda *_a, **_k: SimpleNamespace(status="failed"),
    )
    persistence_failed = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
    )
    assert persistence_failed.errors == ("persistence_failed",)
    assert persistence_failed.previous_artifact_preserved
    assert path.read_bytes() == before


def test_warning_codes_are_sorted_bounded_and_messages_are_safe(tmp_path, monkeypatch):
    warnings = [
        ProjectEvidencePipelineWarning(code=f"warning_{index:03d}", message="DO_NOT_PERSIST_BODY")
        for index in range(150)
    ]
    install_loader(monkeypatch, items=[], warnings=warnings)
    result = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=tmp_path / "memory.json", environ=ENABLED
    )
    codes = [warning.code for warning in result.warnings]
    assert len(codes) == pipeline.MAX_PIPELINE_WARNINGS
    assert codes == sorted(codes)
    assert "DO_NOT_PERSIST_BODY" not in json.dumps(result.to_dict())


def test_result_exposes_only_safe_bounded_metadata(tmp_path, monkeypatch):
    result = run_fixture(monkeypatch, tmp_path, persist=False)
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    for forbidden in (
        "Project memory needed", "atomic persistence", '"allowed_claims"', '"forbidden_claims"',
        '"source_refs"', "backend/memory.py", "raw_text", "full_patch", "Traceback",
    ):
        assert forbidden not in payload
    assert not Path(result.artifact_path or "").is_absolute()
    assert result.next_recommended_action in pipeline.NEXT_RECOMMENDED_ACTIONS


def test_input_order_and_destination_do_not_change_semantic_hash(tmp_path, monkeypatch):
    items = [evidence_input("beta"), evidence_input("alpha")]
    install_loader(monkeypatch, items=items)
    first = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=tmp_path / "one.json",
        persist=False, environ=ENABLED,
    )
    install_loader(monkeypatch, items=reversed(items))
    second = pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=tmp_path / "two.json",
        persist=False, environ={pipeline.PROJECT_EVIDENCE_MEMORY_FLAG: "TRUE"},
    )
    assert first.content_hash == second.content_hash
    assert first.project_count == second.project_count == 2
    assert [stage.to_dict() for stage in first.stages] == [stage.to_dict() for stage in second.stages]


def test_health_disabled_and_missing_do_not_run_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline, "run_project_evidence_pipeline",
        lambda **_kwargs: pytest.fail("pipeline must not run"),
    )
    disabled = pipeline.get_project_evidence_health(
        output_path=tmp_path / "missing.json", environ={}
    )
    missing = pipeline.get_project_evidence_health(
        output_path=tmp_path / "missing.json", environ=ENABLED
    )
    assert disabled.status == "disabled"
    assert missing.status == "missing"


def test_health_ready_or_degraded_is_read_only_and_contains_counts(tmp_path, monkeypatch):
    run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    before = path.read_bytes()
    health = pipeline.get_project_evidence_health(output_path=path, environ=ENABLED)
    assert health.status in {"ready", "degraded"}
    assert health.hash_valid
    assert health.project_count == health.evidence_fact_count == 1
    assert health.capability_fact_count == 0
    assert path.read_bytes() == before
    health_payload = json.dumps(health.to_dict()).casefold()
    assert "allowed_claims" not in health_payload and "forbidden_claims" not in health_payload


def test_health_rejects_hash_mismatch_and_unsupported_schema(tmp_path, monkeypatch):
    run_fixture(monkeypatch, tmp_path)
    path = tmp_path / "project_evidence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content_hash"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert pipeline.get_project_evidence_health(output_path=path, environ=ENABLED).status == "invalid"
    payload["schema_version"] = "project_evidence_memory.v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert pipeline.get_project_evidence_health(output_path=path, environ=ENABLED).status == "invalid"


def test_health_and_inspect_convert_unexpected_load_failure_to_safe_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline.project_memory,
        "load_project_evidence_memory",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("SECRET C:/private")),
    )
    health = pipeline.get_project_evidence_health(output_path=tmp_path / "x.json", environ=ENABLED)
    inspect = pipeline.inspect_project_evidence_memory(output_path=tmp_path / "x.json", environ=ENABLED)
    assert health.status == inspect.status == "error"
    serialized = json.dumps({"health": health.to_dict(), "inspect": inspect.to_dict()})
    assert "SECRET" not in serialized and "C:/private" not in serialized


def test_inspect_is_bounded_sorted_exact_and_content_free(tmp_path, monkeypatch):
    install_loader(monkeypatch, items=[evidence_input("beta"), evidence_input("alpha")])
    path = tmp_path / "memory.json"
    pipeline.run_project_evidence_pipeline(
        source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
    )
    before = path.read_bytes()
    inspected = pipeline.inspect_project_evidence_memory(
        output_path=path, sample_limit=999, environ=ENABLED
    )
    assert inspected.sample_limit == pipeline.MAX_INSPECT_SAMPLE_LIMIT
    assert [item.project_id for item in inspected.projects] == ["alpha", "beta"]
    exact = pipeline.inspect_project_evidence_memory(
        output_path=path, project_id="beta", sample_limit=1, environ=ENABLED
    )
    unknown = pipeline.inspect_project_evidence_memory(
        output_path=path, project_id="Beta", sample_limit=-5, environ=ENABLED
    )
    oversized = pipeline.inspect_project_evidence_memory(
        output_path=path, project_id="x" * 1_000, sample_limit=1, environ=ENABLED
    )
    assert [item.project_id for item in exact.projects] == ["beta"]
    assert unknown.sample_limit == 0 and unknown.projects == () and unknown.project_count == 0
    assert oversized.project_id == "x" * pipeline.MAX_INSPECT_PROJECT_ID_LENGTH
    assert oversized.projects == () and oversized.project_count == 0
    serialized = json.dumps(inspected.to_dict())
    for forbidden in ("mechanism", "implementation", "allowed_claims", "source_refs", "file_path"):
        assert forbidden not in serialized
    assert path.read_bytes() == before


def test_inspect_disabled_missing_and_invalid_are_safe(tmp_path):
    path = tmp_path / "memory.json"
    disabled = pipeline.inspect_project_evidence_memory(output_path=path, environ={})
    missing = pipeline.inspect_project_evidence_memory(output_path=path, environ=ENABLED)
    path.write_text("invalid", encoding="utf-8")
    invalid = pipeline.inspect_project_evidence_memory(output_path=path, environ=ENABLED)
    assert (disabled.status, missing.status, invalid.status) == ("disabled", "missing", "invalid")
    assert invalid.projects == ()


def test_concurrent_builds_serialize_in_process_writes(tmp_path, monkeypatch):
    original_loader = pipeline.input_adapter.load_project_evidence_inputs
    active = 0
    maximum = 0
    guard = threading.Lock()
    records = [evidence_input()]

    def slow_loader(*_args, **_kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return records, []

    monkeypatch.setattr(pipeline.input_adapter, "load_project_evidence_inputs", slow_loader)
    path = tmp_path / "memory.json"

    def run():
        return pipeline.run_project_evidence_pipeline(
            source_paths=DISABLED_PATHS, output_path=path, environ=ENABLED
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _value: run(), range(2)))
    monkeypatch.setattr(pipeline.input_adapter, "load_project_evidence_inputs", original_loader)
    assert maximum == 1
    assert {result.persistence_status for result in results} == {"created", "unchanged"}
    assert load_project_evidence_memory(path).status == "ready"
    artifact_text = path.read_text(encoding="utf-8")
    assert "_BUILD_LOCK" not in artifact_text and "build_lock" not in artifact_text


def test_import_has_no_file_or_external_service_side_effects():
    code = (
        "from pathlib import Path; import sys; "
        "Path.read_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('read')); "
        "Path.write_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('write')); "
        "import backend.project_evidence_pipeline; "
        "assert 'backend.api_server' not in sys.modules; "
        "assert not any(n.startswith(('chromadb','openai','requests','sqlite3')) for n in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_semantic_project_evidence_api_is_the_production_contract():
    root = Path(__file__).resolve().parents[1]
    api_source = (root / "backend" / "api_server.py").read_text(encoding="utf-8")
    for route in ("status", "build", "inspect", "health", "preview", "raw"):
        assert f'/api/project-evidence/{route}' in api_source
    assert "frontend" not in Path(pipeline.__file__).read_text(encoding="utf-8").casefold()


def test_real_enabled_unchanged_disabled_health_and_inspect_smoke(tmp_path):
    accepted = load_project_evidence_memory()
    assert accepted.status == "ready" and accepted.snapshot is not None
    path = tmp_path / "real-project_evidence.json"
    first = pipeline.run_project_evidence_pipeline(
        output_path=path, environ=ENABLED
    )
    first_bytes = path.read_bytes()
    second = pipeline.run_project_evidence_pipeline(
        output_path=path, environ=ENABLED
    )
    disabled = pipeline.run_project_evidence_pipeline(
        output_path=path, environ={}
    )
    health = pipeline.get_project_evidence_health(output_path=path, environ=ENABLED)
    inspected = pipeline.inspect_project_evidence_memory(
        output_path=path, sample_limit=5, environ=ENABLED
    )
    assert first.status in {"ready", "degraded", "unchanged"}
    assert first.project_count == len(accepted.snapshot.projects)
    assert first.evidence_fact_count == sum(
        len(project.evidence_facts) for project in accepted.snapshot.projects
    )
    assert first.capability_fact_count == 0
    assert first.claim_boundary_count == sum(
        len(project.claim_boundaries) for project in accepted.snapshot.projects
    )
    assert first.allowed_claim_count == accepted.snapshot.diagnostics.allowed_claim_count
    assert first.forbidden_claim_count == accepted.snapshot.diagnostics.forbidden_claim_count
    assert first.claim_truncation_count == accepted.snapshot.diagnostics.claim_truncation_count
    assert first.content_hash == accepted.snapshot.content_hash
    assert second.status == "unchanged" and second.content_hash == first.content_hash
    assert disabled.status == "disabled" and path.read_bytes() == first_bytes
    assert health.status in {"ready", "degraded"} and health.hash_valid
    assert inspected.returned_project_count <= 5
    serialized = path.read_text(encoding="utf-8")
    for prohibited in ('"raw_text"', '"full_patch"', '"source_code"', '"api_key"'):
        assert prohibited not in serialized
