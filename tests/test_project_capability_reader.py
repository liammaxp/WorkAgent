from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import backend.project_capability_reader as reader
from backend.project_capability_memory import (
    PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION,
    ProjectCapabilitySourceArtifact,
    build_project_capability_memory,
    load_project_capability_memory,
    persist_project_capability_memory,
)
from backend.project_claim_boundaries import build_project_evidence_claim_boundary
from backend.project_evidence_memory import (
    build_project_evidence_memory_snapshot,
    load_project_evidence_memory,
    serialize_project_evidence_memory_snapshot,
)
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_EVIDENCE = ROOT / "information" / "project_evidence_memory.json"
REAL_CAPABILITY = ROOT / "information" / "project_capability_memory.json"
EXPECTED_CAPABILITY_HASH = "ab588d427e883f5076ea69e04d8bd2f29d9121864a4f743db87ddee874cd5c45"
EXPECTED_EVIDENCE_HASH = "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
EXPECTED_EVIDENCE_FILE_SHA256 = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"


def _evidence_fact(evidence_id: str, project_id: str) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        problem="",
        mechanism="bounded deterministic validation",
        implementation=["Applied schema validation to a bounded result."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{project_id}:{evidence_id}".encode()).hexdigest(),
        )],
        evidence_type=EvidenceType.VALIDATION,
        confidence=Confidence.HIGH,
        technical_tags=["Python"],
        quality_score=90,
        evidence_fact_id=evidence_id,
    )


def _project(
    project_id: str,
    facts: tuple[ProjectEvidenceFact, ...] = (),
) -> ProjectEvidenceMemory:
    boundaries = []
    for fact in facts:
        boundary = build_project_evidence_claim_boundary(fact)
        assert boundary is not None
        boundaries.append(boundary)
    return ProjectEvidenceMemory(
        project_id=project_id,
        project_name=project_id,
        source_hashes={},
        evidence_facts=list(facts),
        capability_facts=[],
        claim_boundaries=boundaries,
        quality_summary={
            "accepted_count": len(facts),
            "supporting_count": 0,
            "weak_count": 0,
            "rejected_count": 0,
        },
    )


def _write_evidence(path: Path) -> None:
    projects = (
        _project("project-a", (_evidence_fact("pef_a", "project-a"),)),
        _project("project-b", (_evidence_fact("pef_b", "project-b"),)),
        _project("project-c"),
    )
    snapshot = build_project_evidence_memory_snapshot(projects)
    path.write_bytes(serialize_project_evidence_memory_snapshot(snapshot))


def _capability_fact(
    project_id: str,
    capability_type: str,
    evidence_id: str,
) -> ProjectCapabilityFact:
    mechanism = f"verified {capability_type.replace('_', ' ')} mechanism"
    return ProjectCapabilityFact(
        project_id=project_id,
        capability_type=capability_type,
        present=True,
        source_evidence_fact_ids=[evidence_id],
        confidence=Confidence.HIGH,
        mechanisms=[mechanism],
        allowed_resume_claims=[f"mechanism:{mechanism}"],
        forbidden_claims=["unsupported metric claim"],
        metric_support=MetricSupport.NONE,
        technical_tags=["Python"],
    )


def _source_artifact(evidence_path: Path) -> ProjectCapabilitySourceArtifact:
    loaded = load_project_evidence_memory(evidence_path)
    assert loaded.status == "ready" and loaded.snapshot is not None
    snapshot = loaded.snapshot
    return ProjectCapabilitySourceArtifact(
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        file_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        project_count=len(snapshot.projects),
        evidence_fact_count=sum(len(project.evidence_facts) for project in snapshot.projects),
        claim_boundary_count=sum(len(project.claim_boundaries) for project in snapshot.projects),
    )


def _write_capability(
    capability_path: Path,
    evidence_path: Path,
    *,
    facts: tuple[ProjectCapabilityFact, ...] = (),
):
    loaded = load_project_evidence_memory(evidence_path)
    assert loaded.status == "ready" and loaded.snapshot is not None
    memory = build_project_capability_memory(
        source_artifact=_source_artifact(evidence_path),
        source_project_ids=tuple(project.project_id for project in loaded.snapshot.projects),
        capability_facts=facts,
    )
    report = persist_project_capability_memory(memory, capability_path)
    assert report.status == "created"
    return memory


def _ready_artifacts(tmp_path: Path):
    evidence_path = tmp_path / "project-evidence.json"
    capability_path = tmp_path / "project-capability.json"
    _write_evidence(evidence_path)
    facts = (
        _capability_fact("project-b", "failure_recovery", "pef_b"),
        _capability_fact("project-a", "output_quality_control", "pef_a"),
        _capability_fact("project-a", "failure_recovery", "pef_a"),
    )
    memory = _write_capability(capability_path, evidence_path, facts=facts)
    return evidence_path, capability_path, memory


def _empty_artifacts(tmp_path: Path):
    evidence_path = tmp_path / "project-evidence.json"
    capability_path = tmp_path / "project-capability.json"
    _write_evidence(evidence_path)
    memory = _write_capability(capability_path, evidence_path)
    return evidence_path, capability_path, memory


def _file_state(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def test_project_capability_reader_import_has_no_side_effects():
    before = _file_state(REAL_CAPABILITY)
    code = (
        "import os; from pathlib import Path; "
        "os.getenv=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected env read')); "
        "Path.read_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected file read')); "
        "Path.write_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected write')); "
        "import backend.project_capability_reader"
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
    assert _file_state(REAL_CAPABILITY) == before


def test_project_capability_reader_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, raising=False)
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("disabled reader must not load capability memory"),
    )
    monkeypatch.setattr(
        reader.evidence_memory_module,
        "load_project_evidence_memory",
        lambda *_args, **_kwargs: pytest.fail("disabled reader must not load evidence memory"),
    )
    result = reader.read_project_capability_memory()
    assert result.status == "disabled"
    assert result.enabled is False
    assert result.facts == ()
    assert result.errors == ()


@pytest.mark.parametrize("value", ("", "0", "false", "FALSE", "no", "off", " Off "))
def test_project_capability_reader_parses_disabled_flag_values(monkeypatch, value):
    monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, value)
    assert reader.is_project_capability_memory_enabled() is False


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", "on", " On "))
def test_project_capability_reader_parses_enabled_flag_values(monkeypatch, value):
    monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, value)
    assert reader.is_project_capability_memory_enabled() is True


def test_project_capability_reader_malformed_flag_does_not_enable_reader(monkeypatch):
    monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, "enable-it")
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("malformed flag must fail before I/O"),
    )
    result = reader.read_project_capability_memory()
    assert result.status == "disabled"
    assert result.enabled is False
    assert result.facts == ()
    assert result.warnings == ("invalid_project_capability_memory_flag",)


def test_reader_returns_successful_empty_result_for_valid_empty_memory(tmp_path):
    evidence_path, capability_path, memory = _empty_artifacts(tmp_path)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "empty"
    assert result.enabled is True
    assert result.project_count == 3
    assert result.capability_fact_count == 0
    assert result.facts == ()
    assert result.artifact_content_hash == memory.content_hash
    assert result.diagnostics["source_lineage_match"] is True


def test_reader_returns_only_authoritative_verified_capability_facts(tmp_path):
    evidence_path, capability_path, memory = _ready_artifacts(tmp_path)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "ready"
    assert result.capability_fact_count == 3
    assert result.facts == memory.capability_facts
    assert all(type(fact) is ProjectCapabilityFact for fact in result.facts)
    assert result.facts is not memory.capability_facts
    assert all(returned is not stored for returned, stored in zip(result.facts, memory.capability_facts))


def test_reader_reports_missing_memory_without_backfill_or_pipeline(
    tmp_path, monkeypatch
):
    import backend.project_capability_backfill as backfill_module
    import backend.project_capability_pipeline as pipeline_module

    evidence_path = tmp_path / "project-evidence.json"
    _write_evidence(evidence_path)
    monkeypatch.setattr(
        backfill_module,
        "run_authoritative_project_capability_backfill",
        lambda: pytest.fail("reader must not backfill"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_project_capability_pipeline",
        lambda **_kwargs: pytest.fail("reader must not run the Pipeline"),
    )
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=tmp_path / "missing.json",
        evidence_memory_path=evidence_path,
    )
    assert result.status == "missing"
    assert result.facts == ()
    assert result.warnings == ("project_capability_memory_missing",)


@pytest.mark.parametrize("mutation", ("malformed", "hash_mismatch"))
def test_reader_rejects_invalid_or_hash_mismatched_memory(tmp_path, mutation):
    evidence_path, capability_path, _memory = _empty_artifacts(tmp_path)
    if mutation == "malformed":
        capability_path.write_text("not json", encoding="utf-8")
    else:
        payload = json.loads(capability_path.read_text(encoding="utf-8"))
        payload["content_hash"] = "0" * 64
        capability_path.write_text(json.dumps(payload), encoding="utf-8")
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "invalid"
    assert result.facts == ()
    assert result.errors == ("project_capability_memory_invalid",)


def test_reader_rejects_unsupported_capability_memory_schema(tmp_path):
    evidence_path, capability_path, _memory = _empty_artifacts(tmp_path)
    payload = json.loads(capability_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "unsupported.capability.memory"
    capability_path.write_text(json.dumps(payload), encoding="utf-8")
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "invalid"
    assert result.facts == ()
    assert result.diagnostics["capability_loader_status"] == "unsupported_version"


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "content_hash",
        "file_sha256",
        "project_count",
        "evidence_fact_count",
        "claim_boundary_count",
    ),
)
def test_reader_rejects_stale_capability_memory_lineage(tmp_path, monkeypatch, field):
    evidence_path, capability_path, memory = _ready_artifacts(tmp_path)
    values = memory.source_artifact.to_dict()
    values[field] = {
        "schema_version": "different.project.evidence.schema",
        "content_hash": "0" * 64,
        "file_sha256": "1" * 64,
        "project_count": values["project_count"] + 1,
        "evidence_fact_count": values["evidence_fact_count"] + 1,
        "claim_boundary_count": values["claim_boundary_count"] + 1,
    }[field]
    fake_source = SimpleNamespace(**values)
    fake_memory = SimpleNamespace(
        schema_version=memory.schema_version,
        content_hash=memory.content_hash,
        source_artifact=fake_source,
        projects=memory.projects,
        capability_facts=memory.capability_facts,
        diagnostics=memory.diagnostics,
    )
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda _path: SimpleNamespace(status="ready", memory=fake_memory),
    )
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    expected_code = f"source_{field}" if field in {"project_count", "evidence_fact_count", "claim_boundary_count"} else f"source_{field}"
    assert result.status == "stale"
    assert result.facts == ()
    assert result.capability_fact_count == 0
    assert expected_code in result.diagnostics["lineage_mismatch_fields"]
    assert "project_capability_memory_stale" in result.warnings


def test_reader_does_not_trust_capability_memory_when_evidence_source_is_missing(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    evidence_path.unlink()
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "stale"
    assert result.facts == ()
    assert "project_evidence_memory_missing" in result.warnings


def test_reader_does_not_trust_capability_memory_when_evidence_source_is_invalid(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    evidence_path.write_text("not json", encoding="utf-8")
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "stale"
    assert result.facts == ()
    assert "project_evidence_memory_invalid" in result.warnings


def test_reader_returns_only_exact_project_capabilities(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    result = reader.get_verified_project_capabilities(
        "project-a",
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "ready"
    assert result.capability_fact_count == 2
    assert {fact.project_id for fact in result.facts} == {"project-a"}
    assert result.diagnostics["selected_project_known"] is True


def test_reader_handles_unknown_project_without_fuzzy_matching(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    for query in ("project", "Project-A", " project-a ", "project-a-extra"):
        result = reader.get_verified_project_capabilities(
            query,
            feature_enabled=True,
            capability_memory_path=capability_path,
            evidence_memory_path=evidence_path,
        )
        assert result.status == "empty"
        assert result.facts == ()
        assert result.diagnostics["selected_project_known"] is False
        assert "project_capability_project_not_found" in result.warnings


def test_reader_distinguishes_known_project_with_zero_capabilities(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    result = reader.get_verified_project_capabilities(
        "project-c",
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert result.status == "empty"
    assert result.facts == ()
    assert result.diagnostics["selected_project_known"] is True
    assert "project_capability_project_has_no_verified_capabilities" in result.warnings
    assert "project_capability_project_not_found" not in result.warnings


def test_reader_never_returns_another_projects_capabilities(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    result = reader.get_verified_project_capabilities(
        "project-b",
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert len(result.facts) == 1
    assert all(fact.project_id == "project-b" for fact in result.facts)
    assert not any(fact.project_id == "project-a" for fact in result.facts)


def test_reader_does_not_fallback_to_candidates_evidence_or_taxonomy(tmp_path):
    evidence_path, capability_path, _memory = _empty_artifacts(tmp_path)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    production_source = (ROOT / "backend" / "project_capability_reader.py").read_text(
        encoding="utf-8"
    )
    assert result.status == "empty"
    assert result.facts == ()
    for forbidden_import in (
        "project_capability_grouping",
        "project_capability_scoring",
        "project_capability_boundaries",
        "project_capability_builder",
        "project_capability_taxonomy",
        "project_capability_pipeline",
        "project_capability_backfill",
    ):
        assert forbidden_import not in production_source


def test_reader_never_runs_project_capability_pipeline(tmp_path, monkeypatch):
    import backend.project_capability_pipeline as pipeline_module

    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "run_project_capability_pipeline",
        lambda **_kwargs: pytest.fail("reader must not invoke Pipeline"),
    )
    assert reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    ).status == "ready"


def test_reader_never_runs_authoritative_backfill(tmp_path, monkeypatch):
    import backend.project_capability_backfill as backfill_module

    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    monkeypatch.setattr(
        backfill_module,
        "run_authoritative_project_capability_backfill",
        lambda: pytest.fail("reader must not invoke backfill"),
    )
    assert reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    ).status == "ready"


def test_reader_is_strictly_read_only(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    before_evidence = _file_state(evidence_path)
    before_capability = _file_state(capability_path)
    for _index in range(2):
        assert reader.read_project_capability_memory(
            feature_enabled=True,
            capability_memory_path=capability_path,
            evidence_memory_path=evidence_path,
        ).status == "ready"
    assert _file_state(evidence_path) == before_evidence
    assert _file_state(capability_path) == before_capability
    assert tuple(tmp_path.iterdir()) == (capability_path, evidence_path) or set(tmp_path.iterdir()) == {
        capability_path,
        evidence_path,
    }


def test_reader_result_does_not_share_mutable_artifact_state(tmp_path):
    evidence_path, capability_path, memory = _ready_artifacts(tmp_path)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "empty"
    with pytest.raises(TypeError):
        result.diagnostics["source_lineage_match"] = False
    with pytest.raises(FrozenInstanceError):
        result.facts[0].project_id = "other-project"
    assert result.facts[0] is not memory.capability_facts[0]
    second = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    assert second == result


def test_reader_fact_ordering_is_deterministic(tmp_path, monkeypatch):
    evidence_path, capability_path, memory = _ready_artifacts(tmp_path)
    reversed_memory = SimpleNamespace(
        schema_version=memory.schema_version,
        content_hash=memory.content_hash,
        source_artifact=memory.source_artifact,
        projects=memory.projects,
        capability_facts=tuple(reversed(memory.capability_facts)),
        diagnostics=memory.diagnostics,
    )
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda _path: SimpleNamespace(status="ready", memory=reversed_memory),
    )
    first = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    second = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    identities = [(fact.project_id, fact.capability_type, fact.capability_id) for fact in first.facts]
    assert identities == sorted(identities)
    assert first == second


def test_reader_diagnostics_are_bounded_and_privacy_safe(tmp_path):
    evidence_path, capability_path, _memory = _ready_artifacts(tmp_path)
    result = reader.read_project_capability_memory(
        feature_enabled=True,
        capability_memory_path=capability_path,
        evidence_memory_path=evidence_path,
    )
    output = json.dumps(result.to_safe_dict(), sort_keys=True).casefold()
    assert len(result.diagnostics) <= reader.MAX_READER_DIAGNOSTICS
    for forbidden in (
        "raw_text",
        "raw_diff",
        "patch",
        "source_code",
        "github_context",
        "authorization",
        "credential",
        "secret",
        "chain_of_thought",
        str(ROOT).casefold(),
    ):
        assert forbidden not in output


def test_reader_maps_unexpected_loader_failure_to_bounded_error(monkeypatch):
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda _path: (_ for _ in ()).throw(RuntimeError("secret private exception")),
    )
    result = reader.read_project_capability_memory(feature_enabled=True)
    assert result.status == "error"
    assert result.facts == ()
    assert result.errors == ("project_capability_reader_error",)
    assert "secret private exception" not in json.dumps(result.to_safe_dict())


def test_real_project_capability_memory_reader_returns_empty_when_enabled():
    before_evidence = _file_state(REAL_EVIDENCE)
    before_capability = _file_state(REAL_CAPABILITY)
    result = reader.read_project_capability_memory(feature_enabled=True)
    assert result.status == "empty"
    assert result.enabled is True
    assert result.project_count == 11
    assert result.capability_fact_count == 0
    assert result.facts == ()
    assert result.artifact_schema_version == PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION
    assert result.artifact_content_hash == EXPECTED_CAPABILITY_HASH
    assert result.source_content_hash == EXPECTED_EVIDENCE_HASH
    assert result.source_file_sha256 == EXPECTED_EVIDENCE_FILE_SHA256
    assert result.diagnostics["source_lineage_match"] is True
    assert _file_state(REAL_EVIDENCE) == before_evidence
    assert _file_state(REAL_CAPABILITY) == before_capability


def test_existing_production_behavior_is_unchanged_when_reader_flag_is_disabled(monkeypatch):
    monkeypatch.setenv(reader.PROJECT_CAPABILITY_MEMORY_FLAG, "0")
    monkeypatch.setattr(
        reader.capability_memory_module,
        "load_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("disabled production reader must not load"),
    )
    assert reader.read_project_capability_memory().status == "disabled"
    consumers = []
    for path in (ROOT / "backend").glob("*.py"):
        if path.name == "project_capability_reader.py":
            continue
        if "project_capability_reader" in path.read_text(encoding="utf-8", errors="ignore"):
            consumers.append(path.name)
    assert consumers == []


def test_reader_step_adds_no_api_or_frontend_surface():
    api_source = (ROOT / "backend" / "api_server.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    frontend_source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "frontend" / "src").rglob("*")
        if path.is_file() and path.suffix.casefold() in {".js", ".jsx", ".ts", ".tsx"}
    )
    for source in (api_source, frontend_source):
        assert "project_capability_reader" not in source
        assert reader.PROJECT_CAPABILITY_MEMORY_FLAG not in source


def test_project_capability_reader_uses_semantic_naming():
    forbidden = (
        "phase" + "5",
        "phase" + "_5",
        "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5",
        "USE_" + "PHASE" + "5",
        "phase" + "5.v1",
    )
    source = (ROOT / "backend" / "project_capability_reader.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert not {item.casefold() for item in forbidden if item.casefold() in source}
    assert reader.PROJECT_CAPABILITY_MEMORY_FLAG == "USE_PROJECT_CAPABILITY_MEMORY"
    assert reader.__name__ == "backend.project_capability_reader"
