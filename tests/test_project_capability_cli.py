from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import backend.project_capability_cli as cli
from backend.project_capability_memory import (
    PROJECT_CAPABILITY_MEMORY_PATH,
    load_project_capability_memory,
    persist_project_capability_memory,
)
from backend.project_capability_pipeline import run_project_capability_pipeline
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_memory import (
    DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
    build_project_evidence_memory_snapshot,
    serialize_project_evidence_memory_snapshot,
)
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceType,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = ROOT / "information" / "project_evidence_memory.json"
REAL_OUTPUT = ROOT / "information" / "project_capability_memory.json"
EXPECTED_SOURCE_CONTENT_HASH = "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
EXPECTED_SOURCE_FILE_HASH = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"


def _fact(
    evidence_id: str,
    *,
    project_id: str = "project-a",
    quality: float = 90,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        problem="",
        mechanism="stable hash-based evidence identity",
        implementation=["Applied deterministic structured validation."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{project_id}:{evidence_id}".encode()).hexdigest(),
        )],
        evidence_type=EvidenceType.VALIDATION,
        confidence=Confidence.HIGH,
        technical_tags=["quality_dimensions", "FastAPI"],
        quality_score=quality,
        evidence_fact_id=evidence_id,
    )


def _project(
    project_id: str,
    facts: tuple[ProjectEvidenceFact, ...] = (),
    *,
    with_boundaries: bool = True,
    source_capabilities: tuple[ProjectCapabilityFact, ...] = (),
) -> ProjectEvidenceMemory:
    boundaries = []
    if with_boundaries:
        for fact in facts:
            boundary = build_project_evidence_claim_boundary(fact)
            assert boundary is not None
            boundaries.append(boundary)
    return ProjectEvidenceMemory(
        project_id=project_id,
        project_name=project_id,
        source_hashes={},
        evidence_facts=list(facts),
        capability_facts=list(source_capabilities),
        claim_boundaries=boundaries,
        quality_summary={
            "accepted_count": len(facts),
            "supporting_count": 0,
            "weak_count": 0,
            "rejected_count": 0,
        },
    )


def _write_source(path: Path, *projects: ProjectEvidenceMemory) -> None:
    snapshot = build_project_evidence_memory_snapshot(projects)
    path.write_bytes(serialize_project_evidence_memory_snapshot(snapshot))


def _empty_source(path: Path) -> None:
    _write_source(path, _project("project-empty"))


def _ready_source(path: Path) -> None:
    fact = _fact("pef_verified")
    _write_source(path, _project("project-a", (fact,)))


def _persist_memory_from_source(source: Path, target: Path):
    result = run_project_capability_pipeline(source_path=source)
    assert result.status in {"ready", "empty"} and result.memory is not None
    report = persist_project_capability_memory(result.memory, target)
    assert report.status == "created"
    return result.memory


def _json_stdout(capsys) -> dict:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def _real_output_state() -> tuple[bool, bytes | None, int | None]:
    if not REAL_OUTPUT.exists():
        return False, None, None
    return True, REAL_OUTPUT.read_bytes(), REAL_OUTPUT.stat().st_mtime_ns


def _backfill_result(status: str = "created"):
    return cli.backfill_module.ProjectCapabilityBackfillResult(
        status=status,
        source_schema_version="project_evidence_memory.v1",
        source_content_hash="1" * 64,
        source_file_sha256_before="2" * 64,
        source_file_sha256_after="2" * 64,
        target_schema_version="project_capability_memory.v1",
        target_content_hash="3" * 64,
        target_file_sha256="4" * 64,
        pipeline_status="empty",
        source_project_count=11,
        source_evidence_fact_count=283,
        source_claim_boundary_count=184,
        source_capability_fact_count=0,
        candidate_count=57,
        assessment_count=57,
        eligible_assessment_count=0,
        policy_count=0,
        build_result_count=0,
        capability_fact_count=0,
        target_existed_before=status == "unchanged",
        target_written=status == "created",
        target_unchanged=status == "unchanged",
        warnings=("capability_facts_empty",),
        errors=(),
        diagnostics={
            "ambiguous_evidence_count": 4,
            "matched_evidence_count": 196,
            "projects_with_capabilities": 0,
            "projects_without_capabilities": 11,
            "skipped_evidence_count": 0,
            "staging_artifact_count": 0,
            "unmatched_evidence_count": 83,
        },
    )


def test_project_capability_cli_import_has_no_side_effects():
    before = _real_output_state()
    code = (
        "from pathlib import Path; import sys; "
        "sys.argv=['project-capability-cli','unexpected']; "
        "Path.read_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected read')); "
        "Path.write_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected write')); "
        "import backend.project_capability_cli; "
        "assert 'backend.api_server' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""
    assert _real_output_state() == before


def test_cli_help_lists_semantic_commands():
    completed = subprocess.run(
        [sys.executable, "-m", "backend.project_capability_cli", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = completed.stdout.casefold()
    assert completed.returncode == 0
    assert all(command in output for command in ("build", "inspect", "validate", "backfill"))
    assert "phase" + "5" not in output
    assert completed.stderr == ""


def test_cli_help_lists_backfill_command():
    completed = subprocess.run(
        [sys.executable, "-m", "backend.project_capability_cli", "backfill", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0
    assert "controlled authoritative backfill" in completed.stdout.casefold()
    assert all(option not in completed.stdout for option in ("--source", "--output", "--force"))
    assert completed.stderr == ""


def test_cli_backfill_calls_dedicated_backfill_api(monkeypatch, capsys):
    calls = []

    def spy():
        calls.append(())
        return _backfill_result()

    monkeypatch.setattr(
        cli.backfill_module,
        "run_authoritative_project_capability_backfill",
        spy,
    )
    assert cli.main(["backfill", "--json"]) == cli.EXIT_SUCCESS
    payload = _json_stdout(capsys)
    assert calls == [()]
    assert payload["command"] == "backfill"
    assert payload["status"] == "created"
    assert payload["target"]["written"] is True


def test_cli_backfill_json_output_is_safe_and_deterministic(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.backfill_module,
        "run_authoritative_project_capability_backfill",
        lambda: _backfill_result(),
    )
    argv = ["backfill", "--json"]
    assert cli.main(argv) == cli.EXIT_SUCCESS
    first = capsys.readouterr().out
    assert cli.main(argv) == cli.EXIT_SUCCESS
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["source"]["project_count"] == 11
    assert payload["pipeline"]["capability_fact_count"] == 0
    assert payload["target"]["staging_artifact_count"] == 0
    assert str(ROOT).casefold() not in first.casefold()
    assert all(
        forbidden not in first.casefold()
        for forbidden in ("raw_text", "raw_diff", "source_code", "credential", "github.com")
    )


@pytest.mark.parametrize("option", ("--source", "--output"))
def test_cli_backfill_does_not_accept_source_or_output_overrides(
    monkeypatch, capsys, option
):
    monkeypatch.setattr(
        cli.backfill_module,
        "run_authoritative_project_capability_backfill",
        lambda: pytest.fail("backfill must not run for invalid arguments"),
    )
    private_value = str(ROOT / "private-override.json")
    assert cli.main(["backfill", option, private_value]) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith("error:invalid_arguments\n")
    assert private_value.casefold() not in captured.err.casefold()


def test_cli_without_command_is_a_usage_error(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.err.endswith("error:invalid_arguments\n")


def test_cli_argument_errors_do_not_echo_private_input(capsys):
    private_input = str(ROOT / "private-secret-value")
    assert cli.main(["inspect", "--unknown", private_input]) == cli.EXIT_USAGE
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err.endswith("error:invalid_arguments\n")
    assert private_input.casefold() not in captured.err.casefold()
    assert "private-secret-value" not in captured.err


def test_cli_build_defaults_to_read_only_pipeline(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    _empty_source(source)
    calls = []
    original = cli.pipeline_module.run_project_capability_pipeline

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cli.pipeline_module, "run_project_capability_pipeline", spy)
    code = cli.main(["build", "--source", str(source), "--json"])
    payload = _json_stdout(capsys)

    assert code == cli.EXIT_SUCCESS
    assert payload["status"] == "empty"
    assert calls == [{"source_path": source, "persist": False, "output_path": None}]
    assert not output.exists()


def test_cli_inspect_uses_existing_pipeline_and_safe_summary(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    _ready_source(source)
    calls = []
    original = cli.pipeline_module.run_project_capability_pipeline

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cli.pipeline_module, "run_project_capability_pipeline", spy)
    assert cli.main(["inspect", "--source", str(source), "--json"]) == 0
    payload = _json_stdout(capsys)

    assert calls == [{"source_path": source, "persist": False, "output_path": None}]
    assert payload["pipeline"]["candidate_count"] == 1
    assert payload["pipeline"]["capability_fact_count"] == 1
    assert "project_results" not in payload


def test_cli_validate_uses_authoritative_memory_loader(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _persist_memory_from_source(source, target)
    calls = []
    original = cli.memory_module.load_project_capability_memory

    def spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(cli.memory_module, "load_project_capability_memory", spy)
    assert cli.main(["validate", "--path", str(target), "--json"]) == 0
    payload = _json_stdout(capsys)

    assert calls == [target]
    assert payload["status"] == "empty"
    assert payload["validation"]["valid"] is True


def test_cli_treats_valid_empty_pipeline_and_memory_as_success(tmp_path, capsys):
    source = tmp_path / "source.json"
    target = tmp_path / "empty.json"
    _empty_source(source)

    assert cli.main(["build", "--source", str(source), "--json"]) == 0
    assert _json_stdout(capsys)["status"] == "empty"
    _persist_memory_from_source(source, target)
    assert cli.main(["validate", "--path", str(target), "--json"]) == 0
    assert _json_stdout(capsys)["status"] == "empty"


def test_cli_treats_ready_pipeline_and_memory_as_success(tmp_path, capsys):
    source = tmp_path / "source.json"
    target = tmp_path / "ready.json"
    _ready_source(source)

    assert cli.main(["build", "--source", str(source), "--json"]) == 0
    assert _json_stdout(capsys)["status"] == "ready"
    _persist_memory_from_source(source, target)
    assert cli.main(["validate", "--path", str(target), "--json"]) == 0
    assert _json_stdout(capsys)["status"] == "ready"


def test_cli_reports_missing_source_with_nonzero_exit_code(tmp_path, capsys):
    code = cli.main(["inspect", "--source", str(tmp_path / "missing.json"), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == cli.EXIT_ARTIFACT_MISSING
    assert payload["status"] == "source_missing"
    assert payload["errors"] == ["source_artifact_missing"]
    assert captured.err == "error:source_artifact_missing\n"


def test_cli_reports_invalid_source_with_nonzero_exit_code(tmp_path, capsys):
    source = tmp_path / "invalid.json"
    source.write_text("not json", encoding="utf-8")
    code = cli.main(["build", "--source", str(source), "--json"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_ARTIFACT_INVALID
    assert json.loads(captured.out)["status"] == "source_invalid"
    assert "traceback" not in captured.err.casefold()


@pytest.mark.parametrize("mutation", ["malformed", "hash_mismatch", "unsupported"])
def test_cli_validate_rejects_invalid_or_hash_mismatched_memory(tmp_path, mutation, capsys):
    source = tmp_path / "source.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    if mutation == "malformed":
        target.write_text("not json", encoding="utf-8")
    else:
        _persist_memory_from_source(source, target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if mutation == "hash_mismatch":
            payload["content_hash"] = "0" * 64
        else:
            payload["schema_version"] = "unsupported.memory.v1"
        target.write_text(json.dumps(payload), encoding="utf-8")

    code = cli.main(["validate", "--path", str(target), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == cli.EXIT_ARTIFACT_INVALID
    assert payload["status"] in {"invalid", "hash_mismatch", "unsupported_version"}
    assert payload["validation"]["valid"] is False
    assert "traceback" not in captured.err.casefold()


def test_cli_validate_reports_missing_artifact(tmp_path, capsys):
    code = cli.main(["validate", "--path", str(tmp_path / "missing.json"), "--json"])
    payload = _json_stdout(capsys)
    assert code == cli.EXIT_ARTIFACT_MISSING
    assert payload["status"] == "missing"
    assert payload["errors"] == ["capability_memory_missing"]


def test_cli_build_persistence_requires_explicit_output_path(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    _empty_source(source)
    monkeypatch.setattr(
        cli.pipeline_module,
        "run_project_capability_pipeline",
        lambda **_kwargs: pytest.fail("pipeline must not run for invalid arguments"),
    )
    code = cli.main(["build", "--source", str(source), "--persist", "--json"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert json.loads(captured.out)["errors"] == ["output_path_required"]
    assert captured.err == "error:output_path_required\n"


def test_cli_rejects_output_path_without_persist(tmp_path, capsys):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    _empty_source(source)
    code = cli.main([
        "build", "--source", str(source), "--output", str(output), "--json"
    ])

    assert code == cli.EXIT_USAGE
    assert _json_stdout(capsys)["errors"] == ["output_path_without_persist"]
    assert not output.exists()


def test_cli_build_can_persist_to_explicit_temporary_path(tmp_path, capsys):
    real_output_before = _real_output_state()
    source = tmp_path / "source.json"
    output = tmp_path / "capability.json"
    _ready_source(source)
    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(output), "--json"
    ])
    payload = _json_stdout(capsys)
    loaded = load_project_capability_memory(output)

    assert code == cli.EXIT_SUCCESS
    assert payload["memory"]["persisted"] is True
    assert payload["memory"]["persisted_path"] == output.name
    assert loaded.status == "ready"
    assert _real_output_state() == real_output_before


def test_generic_cli_build_still_cannot_write_real_capability_path(tmp_path, capsys):
    real_output_before = _real_output_state()
    source = tmp_path / "source.json"
    _empty_source(source)
    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(REAL_OUTPUT), "--json"
    ])

    assert code == cli.EXIT_SAFETY_VIOLATION
    assert _json_stdout(capsys)["errors"] == ["real_backfill_path_prohibited"]
    assert _real_output_state() == real_output_before


def test_cli_cannot_overwrite_project_evidence_memory(tmp_path, capsys):
    source = tmp_path / "source.json"
    _empty_source(source)
    before = REAL_SOURCE.read_bytes()
    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(REAL_SOURCE), "--json"
    ])

    assert code == cli.EXIT_SAFETY_VIOLATION
    assert _json_stdout(capsys)["errors"] == ["evidence_memory_overwrite_prohibited"]
    assert REAL_SOURCE.read_bytes() == before


def test_cli_rejects_source_and_output_path_collision(tmp_path, capsys):
    source = tmp_path / "source.json"
    _empty_source(source)
    before = source.read_bytes()
    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(source), "--json"
    ])

    assert code == cli.EXIT_SAFETY_VIOLATION
    assert _json_stdout(capsys)["errors"] == ["source_output_collision"]
    assert source.read_bytes() == before


def test_cli_rejects_directory_output_path(tmp_path, capsys):
    source = tmp_path / "source.json"
    output = tmp_path / "directory"
    output.mkdir()
    _empty_source(source)

    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(output), "--json"
    ])
    assert code == cli.EXIT_SAFETY_VIOLATION
    assert _json_stdout(capsys)["errors"] == ["output_path_is_directory"]


def test_cli_persistence_failure_preserves_existing_output(tmp_path, capsys):
    source = tmp_path / "source.json"
    output = tmp_path / "capability.json"
    _empty_source(source)
    output.write_bytes(b"preserve-existing-output")
    before = output.read_bytes()

    code = cli.main([
        "build", "--source", str(source), "--persist", "--output", str(output), "--json"
    ])
    captured = capsys.readouterr()

    assert code == cli.EXIT_PERSISTENCE_FAILED
    assert json.loads(captured.out)["errors"] == ["persistence_validation_failed"]
    assert output.read_bytes() == before


def test_cli_json_output_is_deterministic_and_machine_readable(tmp_path, capsys):
    source = tmp_path / "source.json"
    _empty_source(source)
    argv = ["inspect", "--source", str(source), "--json"]

    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert cli.main(argv) == 0
    second = capsys.readouterr().out

    assert first == second
    assert first.endswith("\n") and first.count("\n") == 1
    payload = json.loads(first)
    assert list(sorted(payload)) == sorted((
        "command", "errors", "memory", "pipeline", "source", "status", "warnings"
    ))
    assert "nan" not in first.casefold() and "infinity" not in first.casefold()


def test_cli_human_output_contains_only_safe_summary_fields(tmp_path, capsys):
    source = tmp_path / "source.json"
    _ready_source(source)
    assert cli.main(["inspect", "--source", str(source)]) == 0
    output = capsys.readouterr().out

    assert "status: ready" in output
    assert "pipeline_candidate_count: 1" in output
    assert "source_evidence_fact_count: 1" in output
    assert "pef_verified" not in output
    assert "stable hash-based evidence identity" not in output


def test_cli_output_does_not_expose_raw_or_sensitive_content(tmp_path, capsys):
    source = tmp_path / "source.json"
    _ready_source(source)
    assert cli.main(["inspect", "--source", str(source), "--json"]) == 0
    captured = capsys.readouterr()
    output = (captured.out + captured.err).casefold()

    for forbidden in (
        "raw_text", "raw_diff", "patch", "source_code", "github_context",
        "authorization", "credential", "secret", "chain_of_thought",
        str(ROOT).casefold(),
    ):
        assert forbidden not in output


def test_cli_errors_do_not_emit_tracebacks_or_raw_exception_content(monkeypatch, capsys):
    def fail(**_kwargs):
        raise RuntimeError("raw_text=private-exception-payload")

    monkeypatch.setattr(cli.pipeline_module, "run_project_capability_pipeline", fail)
    code = cli.main(["inspect", "--json"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_PIPELINE_FAILED
    assert json.loads(captured.out)["errors"] == ["pipeline_invocation_failed"]
    assert captured.err == "error:pipeline_invocation_failed\n"
    assert "traceback" not in (captured.out + captured.err).casefold()
    assert "private-exception-payload" not in captured.out + captured.err


def test_cli_build_does_not_copy_source_capability_facts(tmp_path, capsys):
    source = tmp_path / "source.json"
    fact = _fact("pef_source_capability", quality=10)
    source_capability = ProjectCapabilityFact(
        project_id="project-a",
        capability_type="output_quality_control",
        present=True,
        source_evidence_fact_ids=[fact.evidence_fact_id],
        confidence=Confidence.HIGH,
        mechanisms=["legacy producer mechanism"],
        allowed_resume_claims=["mechanism:legacy producer mechanism"],
    )
    _write_source(
        source,
        _project(
            "project-a",
            (fact,),
            with_boundaries=False,
            source_capabilities=(source_capability,),
        ),
    )

    assert cli.main(["build", "--source", str(source), "--json"]) == 0
    payload = _json_stdout(capsys)
    assert payload["source"]["capability_fact_count"] == 1
    assert payload["pipeline"]["capability_fact_count"] == 0
    assert payload["status"] == "empty"
    assert "source_capability_facts_ignored" in payload["warnings"]


def test_cli_calls_existing_pipeline_and_loader_instead_of_lifecycle_internals(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _persist_memory_from_source(source, target)
    pipeline_calls = []
    loader_calls = []
    original_pipeline = cli.pipeline_module.run_project_capability_pipeline
    original_loader = cli.memory_module.load_project_capability_memory

    def pipeline_spy(**kwargs):
        pipeline_calls.append(kwargs)
        return original_pipeline(**kwargs)

    def loader_spy(path):
        loader_calls.append(path)
        return original_loader(path)

    monkeypatch.setattr(cli.pipeline_module, "run_project_capability_pipeline", pipeline_spy)
    monkeypatch.setattr(cli.memory_module, "load_project_capability_memory", loader_spy)
    assert cli.main(["inspect", "--source", str(source), "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["validate", "--path", str(target), "--json"]) == 0
    capsys.readouterr()

    assert len(pipeline_calls) == 1
    assert loader_calls == [target]


def test_cli_tests_do_not_create_real_project_capability_memory(tmp_path, capsys):
    real_output_before = _real_output_state()
    source = tmp_path / "source.json"
    _empty_source(source)
    assert PROJECT_CAPABILITY_MEMORY_PATH == REAL_OUTPUT
    assert cli.main(["build", "--source", str(source), "--json"]) == 0
    capsys.readouterr()
    assert _real_output_state() == real_output_before


def test_cli_inspect_real_evidence_memory_read_only():
    before = REAL_SOURCE.read_bytes()
    real_output_before = _real_output_state()
    completed = subprocess.run(
        [sys.executable, "-m", "backend.project_capability_cli", "inspect", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert payload["status"] == "empty"
    assert payload["source"] == {
        "schema_version": "project_evidence_memory.v1",
        "content_hash": EXPECTED_SOURCE_CONTENT_HASH,
        "file_sha256": EXPECTED_SOURCE_FILE_HASH,
        "project_count": 11,
        "evidence_fact_count": 283,
        "claim_boundary_count": 184,
        "capability_fact_count": 0,
    }
    assert payload["pipeline"]["candidate_count"] == 57
    assert payload["pipeline"]["assessment_count"] == 57
    assert payload["pipeline"]["policy_count"] == 0
    assert payload["pipeline"]["capability_fact_count"] == 0
    assert REAL_SOURCE.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == EXPECTED_SOURCE_FILE_HASH
    assert _real_output_state() == real_output_before


def test_project_capability_cli_uses_semantic_naming():
    forbidden = (
        "phase" + "5",
        "phase" + "_5",
        "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5",
        "USE_" + "PHASE" + "5",
        "phase" + "5.v1",
    )
    source = (ROOT / "backend" / "project_capability_cli.py").read_text(encoding="utf-8").casefold()
    assert not {item.casefold() for item in forbidden if item.casefold() in source}
    assert cli.__name__ == "backend.project_capability_cli"
    assert DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH == REAL_SOURCE
