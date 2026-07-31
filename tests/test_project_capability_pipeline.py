from collections import Counter
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

import backend.project_capability_pipeline as pipeline_module
from backend.project_capability_builder import ProjectCapabilityFactBuildResult
from backend.project_capability_memory import (
    PROJECT_CAPABILITY_MEMORY_PATH,
    PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION,
    load_project_capability_memory,
)
from backend.project_capability_pipeline import (
    ProjectCapabilityPipelineResult,
    build_project_capability_pipeline,
    run_project_capability_pipeline,
)
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
    MetricSupport,
    ProjectCapabilityFact,
    ProjectEvidenceFact,
    ProjectEvidenceMemory,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = ROOT / "information" / "project_evidence_memory.json"
REAL_OUTPUT = ROOT / "information" / "project_capability_memory.json"
EXPECTED_SOURCE_CONTENT_HASH = "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
EXPECTED_SOURCE_FILE_HASH = "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"


def _real_output_state() -> tuple[bool, bytes | None, int | None]:
    if not REAL_OUTPUT.exists():
        return False, None, None
    return True, REAL_OUTPUT.read_bytes(), REAL_OUTPUT.stat().st_mtime_ns


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
        metric_support=MetricSupport.NONE,
        technical_tags=["quality_dimensions", "FastAPI"],
        quality_score=quality,
        evidence_fact_id=evidence_id,
    )


def _project(
    project_id: str,
    facts: tuple[ProjectEvidenceFact, ...] = (),
    *,
    with_boundaries: bool = True,
    source_capability_facts: tuple[ProjectCapabilityFact, ...] = (),
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
        capability_facts=list(source_capability_facts),
        claim_boundaries=boundaries,
        quality_summary={
            "accepted_count": len(facts),
            "supporting_count": 0,
            "weak_count": 0,
            "rejected_count": 0,
        },
    )


def _snapshot(*projects: ProjectEvidenceMemory):
    return build_project_evidence_memory_snapshot(projects)


def _verified_snapshot():
    fact = _fact("pef_verified")
    return _snapshot(_project("project-a", (fact,)))


def _write_source(path: Path, snapshot) -> None:
    path.write_bytes(serialize_project_evidence_memory_snapshot(snapshot))


def test_pipeline_builds_capability_memory_from_verified_synthetic_evidence():
    real_output_before = _real_output_state()
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())

    assert result.status == "ready"
    assert result.source_load_status == "ready"
    assert result.candidate_count == result.assessment_count == 1
    assert result.eligible_assessment_count == 1
    assert result.policy_count == result.eligible_policy_count == 1
    assert result.build_result_count == result.capability_fact_count == 1
    assert result.memory is not None
    assert len(result.memory.capability_facts) == 1
    assert type(result.memory.capability_facts[0]) is ProjectCapabilityFact
    assert result.persisted_path is None
    assert _real_output_state() == real_output_before


def test_pipeline_returns_valid_empty_memory_when_no_capability_is_verified():
    result = build_project_capability_pipeline(
        source_memory=_snapshot(_project("project-empty"))
    )

    assert result.status == "empty"
    assert result.memory is not None
    assert result.memory.capability_facts == ()
    assert result.memory.schema_version == PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION
    assert "capability_facts_empty" in result.warnings
    assert not result.errors


def test_pipeline_preserves_projects_without_verified_capabilities():
    fact = _fact("pef_verified")
    result = build_project_capability_pipeline(
        source_memory=_snapshot(
            _project("project-empty"),
            _project("project-a", (fact,)),
        )
    )

    assert result.status == "ready"
    assert result.memory is not None
    assert tuple(item.project_id for item in result.memory.projects) == ("project-a", "project-empty")
    empty = next(item for item in result.memory.projects if item.project_id == "project-empty")
    assert empty.capability_fact_count == 0


def test_pipeline_accounts_for_every_source_evidence_fact():
    fact = _fact("pef_verified")
    result = build_project_capability_pipeline(source_memory=_snapshot(_project("project-a", (fact,))))

    assert (
        result.matched_evidence_count
        + result.unmatched_evidence_count
        + result.ambiguous_evidence_count
        + result.skipped_evidence_count
        == result.source_evidence_fact_count
    )
    assert all(
        item.matched_evidence_count
        + item.unmatched_evidence_count
        + item.ambiguous_evidence_count
        + item.skipped_evidence_count
        == item.evidence_fact_count
        for item in result.project_results
    )


def test_pipeline_produces_exactly_one_assessment_per_candidate():
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())
    assert result.assessment_count == result.candidate_count
    assert all(item.assessment_count == item.candidate_count for item in result.project_results)


def test_pipeline_fails_closed_when_candidate_lacks_assessment(monkeypatch):
    monkeypatch.setattr(
        pipeline_module.scoring_module,
        "assess_project_capability_candidates",
        lambda **_kwargs: (),
    )
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())

    assert result.status == "failed"
    assert result.memory is None
    assert "candidate_assessment_mismatch" in result.errors


def test_pipeline_does_not_inherit_policies_for_ineligible_assessments():
    fact = _fact("pef_low_quality", quality=10)
    result = build_project_capability_pipeline(
        source_memory=_snapshot(_project("project-a", (fact,), with_boundaries=False))
    )

    assert result.candidate_count == result.assessment_count == 1
    assert result.eligible_assessment_count == 0
    assert result.policy_count == 0
    assert result.status == "empty"


def test_pipeline_does_not_build_fact_from_ineligible_claim_policy():
    fact = _fact("pef_without_boundary")
    result = build_project_capability_pipeline(
        source_memory=_snapshot(_project("project-a", (fact,), with_boundaries=False))
    )

    assert result.eligible_assessment_count == 1
    assert result.policy_count == 1
    assert result.policy_status_counts == {"missing_boundaries": 1}
    assert result.eligible_policy_count == 0
    assert result.build_result_count == result.capability_fact_count == 0
    assert result.status == "empty"


def test_pipeline_fails_closed_when_eligible_assessment_lacks_policy(monkeypatch):
    monkeypatch.setattr(
        pipeline_module.boundary_module,
        "inherit_project_capability_claim_policies",
        lambda **_kwargs: (),
    )
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())

    assert result.status == "failed"
    assert result.memory is None
    assert "eligible_assessment_missing_policy" in result.errors


def test_pipeline_fails_closed_when_eligible_policy_cannot_build_fact(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    output.write_bytes(b"preserve-existing-output")
    before = output.read_bytes()
    _write_source(source, _verified_snapshot())

    def failed_builder(*, project_id, candidates, assessments, policies, evidence_facts):
        candidate = candidates[0]
        policy = policies[0]
        return (ProjectCapabilityFactBuildResult(
            project_id=project_id,
            capability_type=candidate.capability_type,
            build_status="invalid_fact",
            fact=None,
            supporting_evidence_ids=candidate.supporting_evidence_ids,
            inherited_boundary_ids=policy.inherited_boundary_ids,
            reasons=("authoritative_fact_validation_failed",),
            diagnostics={},
        ),)

    monkeypatch.setattr(pipeline_module.builder_module, "build_project_capability_facts", failed_builder)
    result = run_project_capability_pipeline(
        source_path=source, persist=True, output_path=output
    )

    assert result.status == "failed"
    assert result.memory is None
    assert "eligible_policy_build_failed" in result.errors
    assert result.persisted_path is None
    assert output.read_bytes() == before


def test_pipeline_prevents_cross_project_evidence_and_boundary_contamination(monkeypatch):
    original = pipeline_module.grouping_module.group_project_evidence_facts

    def contaminated_grouping(**kwargs):
        return replace(original(**kwargs), project_id="another-project")

    monkeypatch.setattr(
        pipeline_module.grouping_module, "group_project_evidence_facts", contaminated_grouping
    )
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())

    assert result.status == "failed"
    assert result.memory is None
    assert "cross_project_lifecycle_record" in result.errors


def test_pipeline_does_not_copy_source_capability_facts_into_target_memory():
    fact = _fact("pef_source_only", quality=10)
    source_capability = ProjectCapabilityFact(
        project_id="project-a",
        capability_type="output_quality_control",
        present=True,
        source_evidence_fact_ids=[fact.evidence_fact_id],
        confidence=Confidence.HIGH,
        mechanisms=["legacy producer mechanism"],
        allowed_resume_claims=["mechanism:legacy producer mechanism"],
    )
    result = build_project_capability_pipeline(
        source_memory=_snapshot(_project(
            "project-a",
            (fact,),
            with_boundaries=False,
            source_capability_facts=(source_capability,),
        ))
    )

    assert result.source_capability_fact_count == 1
    assert "source_capability_facts_ignored" in result.warnings
    assert result.status == "empty"
    assert result.memory is not None and result.memory.capability_facts == ()


def test_pipeline_preserves_exact_project_evidence_memory_lineage():
    snapshot = _verified_snapshot()
    result = build_project_capability_pipeline(
        source_memory=snapshot, source_file_sha256="a" * 64
    )

    assert result.source_schema_version == snapshot.schema_version
    assert result.source_content_hash == snapshot.content_hash
    assert result.source_file_sha256 == "a" * 64
    assert result.memory is not None
    assert result.memory.source_artifact.to_dict() == {
        "schema_version": snapshot.schema_version,
        "content_hash": snapshot.content_hash,
        "file_sha256": "a" * 64,
        "project_count": 1,
        "evidence_fact_count": 1,
        "claim_boundary_count": 1,
    }


def test_pipeline_reports_missing_source_without_creating_output(tmp_path):
    output = tmp_path / "output.json"
    result = run_project_capability_pipeline(
        source_path=tmp_path / "missing.json", persist=True, output_path=output
    )

    assert result.status == "source_missing"
    assert result.memory is None
    assert result.errors == ("source_artifact_missing",)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["malformed", "hash_mismatch"])
def test_pipeline_rejects_invalid_or_hash_mismatched_source(tmp_path, mutation):
    source = tmp_path / "source.json"
    if mutation == "malformed":
        source.write_text("not json", encoding="utf-8")
    else:
        payload = _verified_snapshot().to_dict()
        payload["content_hash"] = "0" * 64
        source.write_text(json.dumps(payload), encoding="utf-8")

    result = run_project_capability_pipeline(source_path=source)

    assert result.status == "source_invalid"
    assert result.memory is None
    expected = "source_hash_mismatch" if mutation == "hash_mismatch" else "source_artifact_invalid"
    assert result.errors == (expected,)


def test_pipeline_does_not_persist_by_default(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    _write_source(source, _verified_snapshot())

    result = run_project_capability_pipeline(source_path=source, output_path=output)

    assert result.status == "ready"
    assert result.persisted_path is None
    assert not output.exists()


def test_pipeline_requires_explicit_output_path_for_persistence():
    real_output_before = _real_output_state()
    result = run_project_capability_pipeline(source_path=REAL_SOURCE, persist=True)
    assert result.status == "failed"
    assert result.source_load_status == "not_loaded"
    assert result.errors == ("explicit_output_path_required",)
    assert _real_output_state() == real_output_before


def test_pipeline_can_persist_validated_memory_to_explicit_temporary_path(tmp_path):
    real_output_before = _real_output_state()
    source = tmp_path / "source.json"
    output = tmp_path / "project_capability_memory.json"
    _write_source(source, _verified_snapshot())

    result = run_project_capability_pipeline(
        source_path=source, persist=True, output_path=output
    )
    loaded = load_project_capability_memory(output)

    assert result.status == "ready"
    assert result.persisted_path == output.name
    assert loaded.status == "ready"
    assert loaded.memory == result.memory
    assert result.source_file_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert _real_output_state() == real_output_before


def test_pipeline_failure_preserves_existing_output_artifact(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    _write_source(source, _verified_snapshot())
    output.write_bytes(b"invalid-existing-artifact")
    before = output.read_bytes()

    result = run_project_capability_pipeline(
        source_path=source, persist=True, output_path=output
    )

    assert result.status == "failed"
    assert result.memory is None
    assert "persistence_validation_failed" in result.errors
    assert output.read_bytes() == before


def test_pipeline_is_input_order_independent_and_deterministic(tmp_path):
    fact_a = _fact("pef_a", project_id="project-a")
    fact_b = _fact("pef_b", project_id="project-b")
    first = _snapshot(_project("project-b", (fact_b,)), _project("project-a", (fact_a,)))
    second = _snapshot(_project("project-a", (fact_a,)), _project("project-b", (fact_b,)))

    one = build_project_capability_pipeline(source_memory=first, source_file_sha256="a" * 64)
    two = build_project_capability_pipeline(source_memory=second, source_file_sha256="a" * 64)
    assert one == two
    assert one.memory is not None and two.memory is not None
    assert one.memory.content_hash == two.memory.content_hash

    first_path = tmp_path / "one" / "capability.json"
    second_path = tmp_path / "two" / "capability.json"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    pipeline_module.capability_memory_module.persist_project_capability_memory(one.memory, first_path)
    pipeline_module.capability_memory_module.persist_project_capability_memory(two.memory, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_pipeline_does_not_mutate_source_or_lifecycle_objects():
    snapshot = _verified_snapshot()
    before = snapshot.to_dict()
    result = build_project_capability_pipeline(source_memory=snapshot)

    assert snapshot.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"
    with pytest.raises(TypeError):
        result.diagnostics["candidate_count"] = 99
    with pytest.raises(FrozenInstanceError):
        result.project_results[0].status = "failed"


def test_pipeline_diagnostics_are_bounded_and_privacy_safe():
    result = build_project_capability_pipeline(source_memory=_verified_snapshot())
    safe_result = result.to_safe_dict()
    safe_result["memory"] = None
    payload = json.dumps(safe_result, sort_keys=True).casefold()

    assert "stable hash-based evidence identity" not in payload
    assert "applied deterministic structured validation" not in payload
    assert "github_evidence_card" not in payload
    assert "authorization" not in payload
    assert "bearer " not in payload
    assert "raw_patch" not in payload
    assert str(ROOT).casefold() not in payload
    assert set(result.diagnostics) == {
        "eligible_assessment_count",
        "eligible_policy_count",
        "projects_with_capabilities",
        "projects_without_capabilities",
        "source_capability_fact_count",
    }


def test_pipeline_uses_existing_lifecycle_modules(monkeypatch):
    calls = Counter()
    targets = (
        (pipeline_module.grouping_module, "group_project_evidence_facts", "grouping"),
        (pipeline_module.scoring_module, "assess_project_capability_candidates", "scoring"),
        (pipeline_module.boundary_module, "inherit_project_capability_claim_policies", "boundaries"),
        (pipeline_module.builder_module, "build_project_capability_facts", "builder"),
        (pipeline_module.capability_memory_module, "build_project_capability_memory", "memory"),
    )
    for module, name, label in targets:
        original = getattr(module, name)

        def spy(*args, _original=original, _label=label, **kwargs):
            calls[_label] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    result = build_project_capability_pipeline(source_memory=_verified_snapshot())
    assert result.status == "ready"
    assert calls == {"grouping": 1, "scoring": 1, "boundaries": 1, "builder": 1, "memory": 1}


def test_step_does_not_create_real_project_capability_memory_artifact():
    real_output_before = _real_output_state()
    assert PROJECT_CAPABILITY_MEMORY_PATH == REAL_OUTPUT
    result = run_project_capability_pipeline(source_path=REAL_SOURCE)
    assert result.persisted_path is None
    blocked = run_project_capability_pipeline(
        source_path=REAL_SOURCE,
        persist=True,
        output_path=REAL_OUTPUT,
    )
    assert blocked.status == "failed"
    assert blocked.errors == ("real_output_path_forbidden",)
    assert _real_output_state() == real_output_before


def test_pipeline_cannot_overwrite_project_evidence_memory():
    before = REAL_SOURCE.read_bytes()
    result = run_project_capability_pipeline(
        source_path=REAL_SOURCE,
        persist=True,
        output_path=REAL_SOURCE,
    )

    assert result.status == "failed"
    assert result.memory is None
    assert "persistence_validation_failed" in result.errors
    assert REAL_SOURCE.read_bytes() == before


def test_real_project_evidence_memory_pipeline_returns_valid_empty_memory_read_only():
    before_bytes = REAL_SOURCE.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()
    real_output_before = _real_output_state()
    result = run_project_capability_pipeline(source_path=REAL_SOURCE, persist=False)

    assert result.status == "empty"
    assert result.source_project_count == 11
    assert result.source_evidence_fact_count == 283
    assert result.source_claim_boundary_count == 184
    assert result.source_capability_fact_count == 0
    assert result.candidate_count == 57
    assert result.assessment_count == 57
    assert result.eligible_assessment_count == 0
    assert result.policy_count == result.eligible_policy_count == 0
    assert result.build_result_count == result.capability_fact_count == 0
    assert (
        result.matched_evidence_count,
        result.unmatched_evidence_count,
        result.ambiguous_evidence_count,
        result.skipped_evidence_count,
    ) == (196, 83, 4, 0)
    assert result.source_content_hash == EXPECTED_SOURCE_CONTENT_HASH
    assert result.source_file_sha256 == EXPECTED_SOURCE_FILE_HASH == before_hash
    assert result.memory is not None
    assert result.memory.schema_version == PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION
    assert result.memory.diagnostics.projects_without_capabilities == 11
    assert REAL_SOURCE.read_bytes() == before_bytes
    assert hashlib.sha256(REAL_SOURCE.read_bytes()).hexdigest() == before_hash
    assert _real_output_state() == real_output_before


def test_project_capability_pipeline_uses_semantic_naming():
    forbidden = (
        "phase" + "5",
        "phase" + "_5",
        "project_memory_" + "phase" + "5",
        "project_capability_" + "phase" + "5",
        "USE_" + "PHASE" + "5",
        "phase" + "5.v1",
    )
    source = (ROOT / "backend" / "project_capability_pipeline.py").read_text(encoding="utf-8").casefold()
    assert not {token.casefold() for token in forbidden if token.casefold() in source}
    assert DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH == REAL_SOURCE
    assert "project_capability_pipeline" in pipeline_module.__name__
