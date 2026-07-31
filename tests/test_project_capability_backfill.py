from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import backend.project_capability_backfill as backfill
from backend.project_capability_memory import (
    load_project_capability_memory,
    persist_project_capability_memory,
)
from backend.project_capability_pipeline import run_project_capability_pipeline
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
    ProjectCapabilityFact,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_TARGET = ROOT / "information" / "project_capability_memory.json"


def _fact(evidence_id: str, *, project_id: str = "project-a") -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        problem="",
        mechanism="deterministic validation mechanism",
        implementation=["Applied bounded validation."],
        source_refs=[EvidenceSourceRef(
            source_type="github_evidence_card",
            source_id=f"source-{evidence_id}",
            project_id=project_id,
            content_hash=hashlib.sha256(f"{project_id}:{evidence_id}".encode()).hexdigest(),
        )],
        evidence_type=EvidenceType.VALIDATION,
        confidence=Confidence.HIGH,
        technical_tags=["quality_dimensions"],
        quality_score=90,
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


def _empty_source(path: Path, *, project_id: str = "project-empty") -> None:
    _write_source(path, _project(project_id))


def _source_with_capability(path: Path) -> None:
    fact = _fact("pef_source_capability")
    capability = ProjectCapabilityFact(
        project_id="project-a",
        capability_type="output_quality_control",
        present=True,
        source_evidence_fact_ids=[fact.evidence_fact_id],
        confidence=Confidence.HIGH,
        mechanisms=["legacy source mechanism"],
        allowed_resume_claims=["mechanism:legacy source mechanism"],
    )
    _write_source(
        path,
        _project(
            "project-a",
            (fact,),
            with_boundaries=False,
            source_capabilities=(capability,),
        ),
    )


def _configure_paths(monkeypatch, source: Path, target: Path, *, ignored: bool = True) -> None:
    monkeypatch.setattr(backfill, "AUTHORITATIVE_PROJECT_EVIDENCE_MEMORY_PATH", source)
    monkeypatch.setattr(backfill, "AUTHORITATIVE_PROJECT_CAPABILITY_MEMORY_PATH", target)
    monkeypatch.setattr(backfill, "_target_is_git_ignored", lambda path: ignored and path == target)


def _configure_baseline(monkeypatch, source: Path, target: Path):
    _configure_paths(monkeypatch, source, target)
    loaded = load_project_evidence_memory(source)
    assert loaded.status == "ready" and loaded.snapshot is not None
    result = run_project_capability_pipeline(source_path=source, persist=False)
    assert result.status in {"ready", "empty"} and result.memory is not None
    snapshot = loaded.snapshot
    file_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    accepted = {
        "ACCEPTED_SOURCE_SCHEMA_VERSION": snapshot.schema_version,
        "ACCEPTED_SOURCE_CONTENT_HASH": snapshot.content_hash,
        "ACCEPTED_SOURCE_FILE_SHA256": file_sha256,
        "ACCEPTED_SOURCE_PROJECT_COUNT": result.source_project_count,
        "ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT": result.source_evidence_fact_count,
        "ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT": result.source_claim_boundary_count,
        "ACCEPTED_SOURCE_CAPABILITY_FACT_COUNT": result.source_capability_fact_count,
        "ACCEPTED_CANDIDATE_COUNT": result.candidate_count,
        "ACCEPTED_MATCHED_EVIDENCE_COUNT": result.matched_evidence_count,
        "ACCEPTED_UNMATCHED_EVIDENCE_COUNT": result.unmatched_evidence_count,
        "ACCEPTED_AMBIGUOUS_EVIDENCE_COUNT": result.ambiguous_evidence_count,
        "ACCEPTED_SKIPPED_EVIDENCE_COUNT": result.skipped_evidence_count,
        "ACCEPTED_ASSESSMENT_COUNT": result.assessment_count,
        "ACCEPTED_ELIGIBLE_ASSESSMENT_COUNT": result.eligible_assessment_count,
        "ACCEPTED_POLICY_COUNT": result.policy_count,
        "ACCEPTED_BUILD_RESULT_COUNT": result.build_result_count,
        "ACCEPTED_CAPABILITY_FACT_COUNT": result.capability_fact_count,
        "ACCEPTED_PIPELINE_STATUS": result.status,
    }
    for name, value in accepted.items():
        monkeypatch.setattr(backfill, name, value)
    return result


def _real_target_state() -> tuple[bool, bytes | None, int | None]:
    if not REAL_TARGET.exists():
        return False, None, None
    return True, REAL_TARGET.read_bytes(), REAL_TARGET.stat().st_mtime_ns


def test_project_capability_backfill_import_has_no_side_effects():
    before = _real_target_state()
    code = (
        "from pathlib import Path; import subprocess; "
        "Path.read_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected read')); "
        "Path.write_bytes=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected write')); "
        "subprocess.run=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected process')); "
        "import backend.project_capability_backfill"
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
    assert _real_target_state() == before


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "semantic_hash",
        "file_sha256",
        "project_count",
        "evidence_count",
        "boundary_count",
        "source_capability_count",
    ),
)
def test_backfill_requires_exact_authoritative_source_baseline(tmp_path, monkeypatch, mutation):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    if mutation == "source_capability_count":
        _source_with_capability(source)
    else:
        _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    changes = {
        "schema": ("ACCEPTED_SOURCE_SCHEMA_VERSION", "different.schema.v1"),
        "semantic_hash": ("ACCEPTED_SOURCE_CONTENT_HASH", "0" * 64),
        "file_sha256": ("ACCEPTED_SOURCE_FILE_SHA256", "1" * 64),
        "project_count": ("ACCEPTED_SOURCE_PROJECT_COUNT", 2),
        "evidence_count": ("ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT", 1),
        "boundary_count": ("ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT", 1),
        "source_capability_count": ("ACCEPTED_SOURCE_CAPABILITY_FACT_COUNT", 0),
    }
    monkeypatch.setattr(backfill, *changes[mutation])

    result = backfill.run_authoritative_project_capability_backfill()

    assert result.status == "source_baseline_mismatch"
    assert result.errors
    assert not target.exists()


def test_backfill_fails_closed_when_source_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "missing.json"
    target = tmp_path / "capability.json"
    _configure_paths(monkeypatch, source, target)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "source_missing"
    assert result.errors == ("source_artifact_missing",)
    assert not target.exists()


def test_backfill_fails_closed_when_source_is_invalid(tmp_path, monkeypatch):
    source = tmp_path / "invalid.json"
    target = tmp_path / "capability.json"
    source.write_text("not json", encoding="utf-8")
    _configure_paths(monkeypatch, source, target)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "source_invalid"
    assert result.errors == ("source_artifact_invalid",)
    assert not target.exists()


def test_backfill_calls_authoritative_pipeline_read_only(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    calls = []
    original = backfill.pipeline_module.run_project_capability_pipeline

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(backfill.pipeline_module, "run_project_capability_pipeline", spy)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "created"
    assert calls == [{"source_path": source, "persist": False}]


def test_backfill_accepts_valid_empty_pipeline_result(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "created"
    assert result.pipeline_status == "empty"
    assert result.capability_fact_count == 0
    assert "capability_facts_empty" in result.warnings


def test_backfill_rejects_unexpected_real_pipeline_counts(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    expected = _configure_baseline(monkeypatch, source, target)
    unexpected = replace(expected, candidate_count=expected.candidate_count + 1)
    monkeypatch.setattr(
        backfill.pipeline_module,
        "run_project_capability_pipeline",
        lambda **_kwargs: unexpected,
    )
    monkeypatch.setattr(
        backfill.memory_module,
        "persist_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("unexpected persistence"),
    )
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "pipeline_failed"
    assert result.errors == ("pipeline_candidate_count_mismatch",)
    assert not target.exists()


def test_backfill_does_not_persist_failed_pipeline_result(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    expected = _configure_baseline(monkeypatch, source, target)
    failed = replace(expected, status="failed", memory=None, errors=("synthetic_failure",))
    monkeypatch.setattr(
        backfill.pipeline_module,
        "run_project_capability_pipeline",
        lambda **_kwargs: failed,
    )
    monkeypatch.setattr(
        backfill.memory_module,
        "persist_project_capability_memory",
        lambda *_args, **_kwargs: pytest.fail("unexpected persistence"),
    )
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "pipeline_failed"
    assert {"pipeline_status_mismatch", "pipeline_memory_missing"}.issubset(result.errors)
    assert not target.exists()


def test_backfill_atomically_creates_valid_capability_memory(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    expected = _configure_baseline(monkeypatch, source, target)
    calls = []
    original = backfill.memory_module.persist_project_capability_memory

    def spy(memory, path):
        calls.append((memory, path))
        return original(memory, path)

    monkeypatch.setattr(backfill.memory_module, "persist_project_capability_memory", spy)
    result = backfill.run_authoritative_project_capability_backfill()
    loaded = load_project_capability_memory(target)

    assert result.status == "created"
    assert result.target_written is True
    assert calls == [(expected.memory, target)]
    assert loaded.status == "empty"
    assert loaded.memory == expected.memory
    assert not tuple(target.parent.glob(f".{target.name}.*.stage"))


def test_backfill_is_idempotent_and_skips_identical_target(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    first = backfill.run_authoritative_project_capability_backfill()
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    second = backfill.run_authoritative_project_capability_backfill()
    after = (target.read_bytes(), target.stat().st_mtime_ns)

    assert first.status == "created"
    assert second.status == "unchanged"
    assert second.target_written is False
    assert second.target_unchanged is True
    assert second.target_file_sha256 == first.target_file_sha256
    assert second.target_content_hash == first.target_content_hash
    assert after == before


def test_backfill_rejects_valid_but_different_target(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    other_source = tmp_path / "other-evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source, project_id="project-a")
    _empty_source(other_source, project_id="project-b")
    _configure_baseline(monkeypatch, source, target)
    other = run_project_capability_pipeline(source_path=other_source, persist=False)
    assert other.memory is not None
    assert persist_project_capability_memory(other.memory, target).status == "created"
    before = (target.read_bytes(), target.stat().st_mtime_ns)

    result = backfill.run_authoritative_project_capability_backfill()

    assert result.status == "target_conflict"
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_backfill_does_not_overwrite_invalid_target(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    target.write_bytes(b"invalid existing artifact")
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "target_invalid"
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change_stage", ("pipeline", "persistence"))
def test_backfill_fails_if_source_changes_during_execution(
    tmp_path, monkeypatch, change_stage
):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    if change_stage == "pipeline":
        original_pipeline = backfill.pipeline_module.run_project_capability_pipeline

        def change_source_during_pipeline(**kwargs):
            result = original_pipeline(**kwargs)
            source.write_bytes(source.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(
            backfill.pipeline_module,
            "run_project_capability_pipeline",
            change_source_during_pipeline,
        )
    else:
        original_persistence = backfill.memory_module.persist_project_capability_memory

        def change_source_after_persistence(memory, path):
            report = original_persistence(memory, path)
            source.write_bytes(source.read_bytes() + b"\n")
            return report

        monkeypatch.setattr(
            backfill.memory_module,
            "persist_project_capability_memory",
            change_source_after_persistence,
        )
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "verification_failed"
    expected_error = (
        "source_changed_before_persistence"
        if change_stage == "pipeline"
        else "source_changed_after_persistence"
    )
    assert result.errors == (expected_error,)
    assert not target.exists()


def test_backfill_rejects_semantically_equal_noncanonical_target_bytes(
    tmp_path, monkeypatch
):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    expected = _configure_baseline(monkeypatch, source, target)
    assert expected.memory is not None
    assert persist_project_capability_memory(expected.memory, target).status == "created"
    payload = json.loads(target.read_text(encoding="utf-8"))
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    before = (target.read_bytes(), target.stat().st_mtime_ns)

    result = backfill.run_authoritative_project_capability_backfill()

    assert result.status == "target_conflict"
    assert result.errors == ("existing_target_bytes_conflict",)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_backfill_preserves_existing_state_on_persistence_failure(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    sentinel = tmp_path / "sentinel"
    _empty_source(source)
    sentinel.write_bytes(b"preserve")
    _configure_baseline(monkeypatch, source, target)
    monkeypatch.setattr(
        backfill.memory_module,
        "persist_project_capability_memory",
        lambda *_args, **_kwargs: SimpleNamespace(status="failed", round_trip_validated=False),
    )
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "persistence_failed"
    assert not target.exists()
    assert sentinel.read_bytes() == b"preserve"


def test_backfill_rejects_persisted_memory_that_does_not_reload_identically(
    tmp_path, monkeypatch
):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    original = backfill.memory_module.persist_project_capability_memory

    def persist_then_corrupt(memory, path):
        report = original(memory, path)
        Path(path).write_bytes(b"{}")
        return report

    monkeypatch.setattr(
        backfill.memory_module,
        "persist_project_capability_memory",
        persist_then_corrupt,
    )
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "verification_failed"
    assert result.errors == ("persisted_target_invalid",)


def test_backfill_never_modifies_project_evidence_memory(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "created"
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_backfill_does_not_bypass_pipeline_or_call_lifecycle_stages_directly(
    tmp_path, monkeypatch
):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    calls = []
    original = backfill.pipeline_module.run_project_capability_pipeline

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(backfill.pipeline_module, "run_project_capability_pipeline", spy)
    result = backfill.run_authoritative_project_capability_backfill()
    production_source = (ROOT / "backend" / "project_capability_backfill.py").read_text(
        encoding="utf-8"
    )
    assert result.status == "created"
    assert calls == [{"source_path": source, "persist": False}]
    for module_name in (
        "project_capability_grouping",
        "project_capability_scoring",
        "project_capability_boundaries",
        "project_capability_builder",
    ):
        assert module_name not in production_source


def test_backfill_fails_closed_when_target_is_not_git_ignored(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_paths(monkeypatch, source, target, ignored=False)
    result = backfill.run_authoritative_project_capability_backfill()
    assert result.status == "verification_failed"
    assert result.errors == ("target_not_git_ignored",)
    assert not target.exists()


def test_backfill_result_contains_no_raw_or_sensitive_content(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    result = backfill.run_authoritative_project_capability_backfill()
    serialized = json.dumps(result.to_safe_dict(), sort_keys=True).casefold()
    for forbidden in (
        "raw_text",
        "raw_diff",
        "patch",
        "source_code",
        "authorization",
        "credential",
        "secret",
        "github.com",
        str(ROOT).casefold(),
    ):
        assert forbidden not in serialized


def test_backfill_tests_do_not_create_real_project_capability_memory(tmp_path, monkeypatch):
    before = _real_target_state()
    source = tmp_path / "evidence.json"
    target = tmp_path / "capability.json"
    _empty_source(source)
    _configure_baseline(monkeypatch, source, target)
    assert backfill.run_authoritative_project_capability_backfill().status == "created"
    assert target.exists()
    assert _real_target_state() == before


def test_project_capability_backfill_uses_semantic_naming():
    forbidden = (
        "phase" + "5",
        "phase" + "_5",
        "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5",
        "USE_" + "PHASE" + "5",
        "phase" + "5.v1",
    )
    source = (ROOT / "backend" / "project_capability_backfill.py").read_text(
        encoding="utf-8"
    ).casefold()
    assert not {item.casefold() for item in forbidden if item.casefold() in source}
    assert backfill.__name__ == "backend.project_capability_backfill"
