from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import pytest

from backend.phase4_capability_extractor import extract_phase4_capabilities_by_project
from backend.phase4_claim_boundary import (
    CLAIM_LIMITS,
    CLAIM_TYPES,
    GLOBAL_FORBIDDEN_POLICIES,
    MAX_DECISION_SAMPLES,
    Phase4StructuredClaim,
    build_phase4_capability_claim_boundary,
    build_phase4_claim_boundaries_by_project,
    build_phase4_evidence_claim_boundary,
    build_phase4_project_claim_boundary,
    list_phase4_claim_types,
    validate_phase4_claim_boundary,
)
from backend.phase4_evidence_normalizer import dedupe_phase4_inputs
from backend.phase4_evidence_scoring import score_phase4_evidence_facts
from backend.phase4_evidence_synthesizer import synthesize_phase4_evidence_facts
from backend.phase4_input_adapter import load_phase4_inputs
from backend.phase4_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    Phase4CapabilityFact,
    Phase4ClaimBoundary,
    Phase4EvidenceFact,
    Phase4SourceRef,
)


def source(
    source_id: str = "src-1",
    *,
    project_id: str = "workagent",
    source_type: str = "phase3_evidence_card",
    file_path: str | None = "backend/module.py",
    symbol: str | None = "build_result",
) -> Phase4SourceRef:
    return Phase4SourceRef(
        source_type=source_type,
        source_id=source_id,
        project_id=project_id,
        content_hash=hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
        file_path=file_path,
        symbol=symbol,
    )


def fact(
    mechanism: str = "deterministic ordering",
    *,
    project_id: str = "workagent",
    source_id: str = "src-1",
    source_type: str = "phase3_evidence_card",
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    quality_score: int = 80,
    problem: str = "Evidence order varied across runs.",
    implementation: list[str] | None = None,
    safe_impact: list[str] | None = None,
    technical_tags: list[str] | None = None,
    metric_support: MetricSupport = MetricSupport.NONE,
    forbidden_claims: list[str] | None = None,
    evidence_fact_id: str = "",
    blocker_codes: list[str] | None = None,
) -> Phase4EvidenceFact:
    return Phase4EvidenceFact(
        project_id=project_id,
        mechanism=mechanism,
        implementation=["backend/module.py::build_result"] if implementation is None else implementation,
        problem=problem,
        safe_impact=[] if safe_impact is None else safe_impact,
        technical_tags=[] if technical_tags is None else technical_tags,
        evidence_type=EvidenceType.ARCHITECTURE,
        confidence=Confidence.MEDIUM,
        metric_support=metric_support,
        forbidden_claims=[] if forbidden_claims is None else forbidden_claims,
        status=status,
        quality_score=quality_score,
        quality_breakdown={
            "recommended_status": status.value,
            "recommended_quality_band": "high_value" if quality_score >= 80 else "supporting_value",
            "blocker_codes": [] if blocker_codes is None else blocker_codes,
        },
        source_refs=[source(source_id, project_id=project_id, source_type=source_type)],
        evidence_fact_id=evidence_fact_id,
    )


def capability(
    evidence: Phase4EvidenceFact,
    *,
    capability_type: str = "deterministic_evidence_normalization",
    present: bool = True,
    project_id: str | None = None,
) -> Phase4CapabilityFact:
    return Phase4CapabilityFact(
        project_id=project_id or evidence.project_id,
        capability_type=capability_type,
        present=present,
        source_evidence_fact_ids=[evidence.evidence_fact_id] if present else [],
        confidence=Confidence.HIGH,
    )


def allowed(boundary: Phase4ClaimBoundary | None, prefix: str) -> list[str]:
    assert boundary is not None
    return [value for value in boundary.allowed_claims if value.startswith(prefix + ":")]


def metadata_note(boundary: Phase4ClaimBoundary, prefix: str) -> str:
    claim_value = next(value for value in boundary.allowed_claims if value.startswith(prefix + ":"))
    claim_hash = hashlib.sha256(claim_value.encode("utf-8")).hexdigest()[:16]
    return next(note for note in boundary.notes if note.startswith(f"claim_meta|{claim_hash}|"))


@pytest.mark.parametrize("quality", [80, 90, 100])
def test_high_quality_direct_fact_creates_mechanism_claim(quality):
    boundary = build_phase4_evidence_claim_boundary(fact(quality_score=quality))
    assert allowed(boundary, "mechanism") == ["mechanism:deterministic ordering"]
    assert "|high|" in metadata_note(boundary, "mechanism")


@pytest.mark.parametrize("quality", [60, 61, 70, 79])
def test_medium_quality_direct_fact_creates_bounded_mechanism_claim(quality):
    boundary = build_phase4_evidence_claim_boundary(fact(quality_score=quality))
    assert allowed(boundary, "mechanism")
    assert "|medium|" in metadata_note(boundary, "mechanism")


@pytest.mark.parametrize("quality", [0, 39, 40, 59])
def test_low_quality_direct_fact_creates_no_affirmative_claim(quality):
    assert build_phase4_evidence_claim_boundary(fact(quality_score=quality)) is None


@pytest.mark.parametrize("status", [EvidenceStatus.WEAK, EvidenceStatus.REJECTED])
def test_weak_or_rejected_fact_creates_no_affirmative_claim(status):
    assert build_phase4_evidence_claim_boundary(fact(status=status)) is None


def test_supporting_project_memory_fact_creates_only_bounded_problem_context():
    item = fact(
        source_type="project_memory",
        status=EvidenceStatus.SUPPORTING,
        problem="The project tracks local evidence state.",
        technical_tags=["Python"],
    )
    boundary = build_phase4_evidence_claim_boundary(item)
    assert allowed(boundary, "problem")
    assert not allowed(boundary, "mechanism")
    assert not allowed(boundary, "implementation")
    assert not allowed(boundary, "technology")
    assert "|low|" in metadata_note(boundary, "problem")


def test_capability_context_weak_fact_creates_no_allowed_claim():
    item = fact(source_type="phase2_capability_fact", status=EvidenceStatus.WEAK)
    assert build_phase4_evidence_claim_boundary(item) is None


def test_missing_provenance_blocks_claims():
    item = fact()
    item.source_refs.clear()
    assert build_phase4_evidence_claim_boundary(item) is None


def test_project_mismatched_provenance_blocks_claims():
    item = fact()
    item.source_refs[0] = source("other", project_id="other")
    assert build_phase4_evidence_claim_boundary(item) is None


def test_explicit_problem_is_preserved_without_paraphrase():
    item = fact(problem="  Exact   bounded problem. ")
    boundary = build_phase4_evidence_claim_boundary(item)
    assert allowed(boundary, "problem") == ["problem:Exact bounded problem."]


def test_missing_problem_is_not_inferred_from_mechanism():
    boundary = build_phase4_evidence_claim_boundary(fact(problem=""))
    assert not allowed(boundary, "problem")
    assert allowed(boundary, "mechanism")


@pytest.mark.parametrize(
    "mechanism",
    [
        "updated backend logic",
        "added validation",
        "improved project workflow",
        "improved pipeline",
        "optimized performance",
        "unknown",
    ],
)
def test_generic_mechanism_is_blocked(mechanism):
    assert build_phase4_evidence_claim_boundary(fact(mechanism, problem="")) is None


def test_mechanism_does_not_create_impact_or_capability_claim():
    boundary = build_phase4_evidence_claim_boundary(fact("deterministic ordering", problem=""))
    assert allowed(boundary, "mechanism")
    assert not allowed(boundary, "impact")
    assert not allowed(boundary, "capability")


@pytest.mark.parametrize(
    ("implementation", "expected"),
    [
        ("backend/module.py", "implementation:backend/module.py"),
        ("backend/module.py::build_result", "implementation:backend/module.py::build_result"),
        ("POST /api/evidence/build", "implementation:POST /api/evidence/build"),
        ("C:/Users/private/repo/backend/module.py::build_result", "implementation:backend/module.py::build_result"),
        ("/home/private/repo/frontend/src/api.ts", "implementation:frontend/src/api.ts"),
    ],
)
def test_explicit_implementation_is_safely_normalized(implementation, expected):
    boundary = build_phase4_evidence_claim_boundary(fact(implementation=[implementation]))
    assert expected in allowed(boundary, "implementation")
    assert not allowed(boundary, "architecture")


def test_implementation_claim_retains_evidence_support_id():
    item = fact(evidence_fact_id="p4ef_explicit_support")
    boundary = build_phase4_evidence_claim_boundary(item)
    assert item.evidence_fact_id in metadata_note(boundary, "implementation")


def test_exact_duplicate_implementation_claims_merge_support_ids():
    first = fact("deterministic ordering", source_id="a", implementation=["backend/shared.py"])
    second = fact("canonical serialization", source_id="b", implementation=["backend/shared.py"])
    boundary = build_phase4_project_claim_boundary("workagent", [first, second])
    assert allowed(boundary, "implementation") == ["implementation:backend/shared.py"]
    note = metadata_note(boundary, "implementation")
    assert first.evidence_fact_id in note and second.evidence_fact_id in note


def test_similar_implementation_claims_remain_separate():
    first = fact(source_id="a", implementation=["backend/validate_output.py"])
    second = fact(source_id="b", implementation=["backend/validate_claim.py"])
    boundary = build_phase4_project_claim_boundary("workagent", [first, second])
    assert len(allowed(boundary, "implementation")) == 2


def test_explicit_safe_impact_creates_bounded_impact_only():
    boundary = build_phase4_evidence_claim_boundary(fact(safe_impact=["Kept evidence ordering stable."]))
    assert allowed(boundary, "impact") == ["impact:Kept evidence ordering stable."]
    assert not allowed(boundary, "metric")


@pytest.mark.parametrize(
    ("impact", "policy"),
    [
        ("Improved reliability", "unsupported_reliability_improvement"),
        ("Eliminated hallucinations", "hallucination_elimination"),
        ("Reduced token usage", "unsupported_token_reduction"),
        ("Improved ATS score", "unsupported_ats_improvement"),
        ("Improved performance", "unsupported_performance_improvement"),
        ("Guaranteed correctness", "absolute_guarantee"),
        ("Provided production scale", "unsupported_production_scale"),
        ("Delivered enterprise grade output", "unsupported_enterprise_grade"),
        ("Completed production deployment", "unsupported_deployment_claim"),
    ],
)
def test_unsafe_impact_is_blocked_without_removing_mechanism(impact, policy):
    boundary = build_phase4_evidence_claim_boundary(fact(safe_impact=[impact]))
    assert not allowed(boundary, "impact")
    assert allowed(boundary, "mechanism")
    assert policy in boundary.forbidden_claims


def test_technical_tag_with_concrete_implementation_creates_project_local_technology():
    item = fact(technical_tags=["Python"], implementation=["backend/module.py"])
    boundary = build_phase4_evidence_claim_boundary(item)
    assert allowed(boundary, "technology") == ["technology:Python"]


def test_technical_tag_without_concrete_implementation_creates_no_technology():
    item = fact(technical_tags=["Python"], implementation=["updated code"])
    boundary = build_phase4_evidence_claim_boundary(item)
    assert boundary is not None
    assert not allowed(boundary, "technology")


@pytest.mark.parametrize(
    ("source_type", "status"),
    [
        ("project_memory", EvidenceStatus.SUPPORTING),
        ("phase2_capability_fact", EvidenceStatus.WEAK),
    ],
)
def test_contextual_technology_never_becomes_allowed(source_type, status):
    item = fact(source_type=source_type, status=status, technical_tags=["Python"])
    boundary = build_phase4_evidence_claim_boundary(item)
    assert boundary is None or not allowed(boundary, "technology")


def test_technology_cannot_leak_between_projects():
    first = fact(project_id="alpha", source_id="a", technical_tags=["Python"])
    second = fact(project_id="beta", source_id="b", technical_tags=["Rust"])
    boundaries, _ = build_phase4_claim_boundaries_by_project([first, second])
    assert allowed(boundaries["alpha"], "technology") == ["technology:Python"]
    assert allowed(boundaries["beta"], "technology") == ["technology:Rust"]


def test_builder_does_not_read_jd_resume_or_global_ontology(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("unexpected file read"))
    boundary = build_phase4_evidence_claim_boundary(fact(technical_tags=["Python"]))
    assert allowed(boundary, "technology")


def test_present_capability_fact_creates_canonical_capability_claim_and_taxonomy_policies():
    evidence = fact()
    cap = capability(evidence)
    boundary = build_phase4_capability_claim_boundary(cap, evidence_facts_by_id={evidence.evidence_fact_id: evidence})
    assert allowed(boundary, "capability") == ["capability:deterministic_evidence_normalization"]
    assert any(value.startswith("taxonomy:") for value in boundary.forbidden_claims)
    assert cap.capability_id in metadata_note(boundary, "capability")


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("deterministic_latex_validation", "latex_validation_and_repair"),
        ("local_project_memory", "project_memory_management"),
        ("testing_and_regression_safety", "test_and_regression_hardening"),
        ("token_or_cost_reduction", "token_or_context_efficiency"),
        ("unsupported_claim_boundary", "claim_validation"),
    ],
)
def test_declared_capability_alias_resolves_only_with_present_fact(alias, canonical):
    evidence = fact()
    cap = capability(evidence, capability_type=alias)
    boundary = build_phase4_capability_claim_boundary(cap, evidence_facts_by_id={evidence.evidence_fact_id: evidence})
    assert allowed(boundary, "capability") == [f"capability:{canonical}"]


def test_absent_capability_fact_creates_no_boundary():
    evidence = fact()
    assert build_phase4_capability_claim_boundary(
        capability(evidence, present=False), evidence_facts_by_id={evidence.evidence_fact_id: evidence}
    ) is None


def test_missing_or_cross_project_capability_evidence_blocks_claim():
    evidence = fact()
    cap = capability(evidence)
    assert build_phase4_capability_claim_boundary(cap, evidence_facts_by_id={}) is None
    other = fact(project_id="other", evidence_fact_id=evidence.evidence_fact_id)
    assert build_phase4_capability_claim_boundary(
        cap, evidence_facts_by_id={evidence.evidence_fact_id: other}
    ) is None


def test_zero_capabilities_still_produces_project_boundary():
    boundary = build_phase4_project_claim_boundary("workagent", [fact()], [])
    assert boundary is not None
    assert boundary.subject_type is ClaimSubjectType.PROJECT
    assert not allowed(boundary, "capability")


def test_explicit_metric_support_creates_metric_claim():
    boundary = build_phase4_evidence_claim_boundary(
        fact(safe_impact=["Reduced processing time by 25%"], metric_support=MetricSupport.EXPLICIT)
    )
    assert allowed(boundary, "metric")
    assert boundary.metric_support is MetricSupport.EXPLICIT


@pytest.mark.parametrize("impact", ["Approximately 25% lower latency", "About 2x throughput", "20-30% fewer retries"])
def test_approximate_metric_requires_approximate_or_range_wording(impact):
    boundary = build_phase4_evidence_claim_boundary(
        fact(safe_impact=[impact], metric_support=MetricSupport.APPROXIMATE)
    )
    assert allowed(boundary, "metric")
    assert boundary.metric_support is MetricSupport.APPROXIMATE


def test_approximate_support_blocks_unqualified_exact_wording():
    boundary = build_phase4_evidence_claim_boundary(
        fact(safe_impact=["Reduced processing time by 25%"], metric_support=MetricSupport.APPROXIMATE)
    )
    assert not allowed(boundary, "metric")
    assert "unsupported_metric" in boundary.forbidden_claims


def test_metric_support_none_blocks_numeric_impact():
    boundary = build_phase4_evidence_claim_boundary(fact(safe_impact=["Reduced processing time by 25%"]))
    assert not allowed(boundary, "metric")
    assert not allowed(boundary, "impact")
    assert "unsupported_metric" in boundary.forbidden_claims


@pytest.mark.parametrize(
    "value",
    [
        "Quality score 92",
        "Added 12 tests",
        "Changed 200 lines",
        "Updated 4 files",
        "Schema version 4",
        "Handled HTTP status 404",
        "Model version 3",
        "Commit abcdef1234567890",
        "Completed on 2026-07-27",
    ],
)
def test_operational_counts_and_identifiers_are_not_resume_metrics(value):
    boundary = build_phase4_evidence_claim_boundary(
        fact(safe_impact=[value], metric_support=MetricSupport.EXPLICIT)
    )
    assert not allowed(boundary, "metric")


def test_conflicting_metric_support_resolves_conservatively():
    explicit = fact(
        source_id="explicit",
        safe_impact=["Reduced processing time by 25%"],
        metric_support=MetricSupport.EXPLICIT,
    )
    unsupported = fact(source_id="none", safe_impact=["Reduced processing time by 25%"])
    boundary = build_phase4_project_claim_boundary("workagent", [explicit, unsupported])
    assert not allowed(boundary, "metric")
    assert not allowed(boundary, "impact")


@pytest.mark.parametrize(
    "policy",
    [
        "unsupported_metric",
        "absolute_guarantee",
        "hallucination_elimination",
        "unsupported_token_reduction",
        "cross_project_technology",
        "legacy_capability_label_promotion",
        "mechanism_to_impact_inference",
        "unsupported_production_scale",
        "unsupported_enterprise_grade",
        "unsupported_deployment_claim",
        "unsupported_capability",
        "unsupported_architecture",
    ],
)
def test_global_forbidden_policy_is_present(policy):
    boundary = build_phase4_evidence_claim_boundary(fact())
    assert policy in boundary.forbidden_claims


def test_forbidden_claims_are_deterministic_and_deduplicated():
    first = build_phase4_evidence_claim_boundary(fact())
    second = build_phase4_evidence_claim_boundary(fact())
    assert first.forbidden_claims == sorted(set(first.forbidden_claims), key=lambda value: (value.casefold(), value))
    assert first.forbidden_claims == second.forbidden_claims


def test_same_project_aggregates_and_different_projects_remain_exactly_separate():
    records = [
        fact(project_id="workagent", source_id="a"),
        fact(project_id="example/workagent", source_id="b"),
        fact(project_id="liammaxp/WorkAgent", source_id="c"),
    ]
    boundaries, report = build_phase4_claim_boundaries_by_project(reversed(records))
    assert list(boundaries) == ["example/workagent", "liammaxp/WorkAgent", "workagent"]
    assert report.project_count == 3
    assert all(boundary.project_id == key for key, boundary in boundaries.items())


def test_project_boundary_contains_only_support_from_its_project():
    alpha = fact(project_id="alpha", source_id="a")
    beta = fact(project_id="beta", source_id="b")
    boundaries, _ = build_phase4_claim_boundaries_by_project([alpha, beta])
    assert beta.evidence_fact_id not in "\n".join(boundaries["alpha"].notes)
    assert alpha.evidence_fact_id not in "\n".join(boundaries["beta"].notes)


def test_empty_or_nonqualifying_project_has_documented_empty_result():
    assert build_phase4_project_claim_boundary("workagent", []) is None
    assert build_phase4_project_claim_boundary("workagent", [fact(status=EvidenceStatus.WEAK)]) is None


def test_forbidden_wins_on_exact_hard_conflict():
    item = fact(forbidden_claims=["mechanism:deterministic ordering"])
    boundary = build_phase4_evidence_claim_boundary(item)
    assert boundary is not None
    assert "mechanism:deterministic ordering" not in boundary.allowed_claims
    assert "mechanism:deterministic ordering" in boundary.forbidden_claims


def test_duplicate_support_does_not_raise_claim_confidence():
    high = fact(source_id="high", quality_score=80)
    medium = fact(source_id="medium", quality_score=60)
    boundary = build_phase4_project_claim_boundary("workagent", [medium, high])
    assert "|high|" in metadata_note(boundary, "mechanism")
    assert build_phase4_project_claim_boundary("workagent", [high]).boundary_id != ""


def test_legacy_label_does_not_inflate_confidence_or_become_allowed():
    direct = fact(source_id="direct", quality_score=60)
    legacy = fact(
        source_id="legacy",
        source_type="phase2_capability_fact",
        status=EvidenceStatus.WEAK,
        technical_tags=["local_project_memory"],
    )
    boundary = build_phase4_project_claim_boundary("workagent", [direct, legacy])
    assert "|medium|" in metadata_note(boundary, "mechanism")
    assert not allowed(boundary, "capability")
    assert "unsupported_capability:project_memory_management" in boundary.forbidden_claims


def test_structured_claim_registry_is_immutable_and_validated():
    assert list_phase4_claim_types() == CLAIM_TYPES
    assert len(CLAIM_TYPES) == len(set(CLAIM_TYPES))
    with pytest.raises(ValueError, match="unknown claim type"):
        Phase4StructuredClaim(
            project_id="workagent",
            claim_type="invented",
            value="value",
            evidence_fact_ids=(),
            capability_fact_ids=(),
            metric_support="none",
            confidence="low",
            rule_id="test_rule",
        )


@pytest.mark.parametrize(
    ("claim", "error"),
    [
        ("unknown:value", "unknown_claim_type"),
        ("Not Valid:value", "invalid_claim_prefix"),
        ("mechanism:", "blank_claim_value"),
        ("implementation:C:/Users/private/file.py", "unsafe_absolute_path"),
        ("mechanism:diff --git a/x b/x", "unsafe_claim_content"),
        ("mechanism:raw_text=complete private body", "unsafe_claim_content"),
        ("mechanism:def run():", "unsafe_claim_content"),
        ("metric:25% faster", "unsupported_metric_claim"),
    ],
)
def test_validation_rejects_invalid_or_unsafe_claims(claim, error):
    boundary = Phase4ClaimBoundary(
        project_id="workagent",
        subject_type=ClaimSubjectType.PROJECT,
        subject_id="workagent",
        allowed_claims=[claim],
    )
    result = validate_phase4_claim_boundary(boundary)
    assert not result.valid
    assert error in result.errors


def test_validation_detects_same_claim_allowed_and_forbidden_after_mutation():
    boundary = build_phase4_evidence_claim_boundary(fact())
    boundary.forbidden_claims.append(boundary.allowed_claims[0])
    result = validate_phase4_claim_boundary(boundary)
    assert "allowed_forbidden_conflict" in result.errors


def test_validation_detects_unknown_evidence_and_capability_ids():
    evidence = fact()
    cap = capability(evidence)
    boundary = build_phase4_capability_claim_boundary(cap, evidence_facts_by_id={evidence.evidence_fact_id: evidence})
    result = validate_phase4_claim_boundary(boundary, evidence_facts_by_id={}, capability_facts_by_id={})
    assert "unknown_evidence_fact_id" in result.errors
    assert "unknown_capability_fact_id" in result.errors


def test_validation_errors_never_include_claim_body():
    secret_body = "api_key=super-secret-value"
    boundary = Phase4ClaimBoundary(
        project_id="workagent",
        subject_type=ClaimSubjectType.PROJECT,
        subject_id="workagent",
        allowed_claims=[f"mechanism:{secret_body}"],
    )
    result = validate_phase4_claim_boundary(boundary)
    assert not result.valid
    assert secret_body not in " ".join(result.errors)


def test_inputs_are_not_mutated():
    item = fact(technical_tags=["Python", "FastAPI"])
    before = item.to_json()
    build_phase4_evidence_claim_boundary(item)
    build_phase4_project_claim_boundary(item.project_id, [item])
    assert item.to_json() == before


def test_equivalent_evidence_is_idempotent_and_material_claim_changes_change_id():
    first = build_phase4_evidence_claim_boundary(fact())
    second = build_phase4_evidence_claim_boundary(fact())
    changed_allowed = build_phase4_evidence_claim_boundary(fact(mechanism="canonical serialization"))
    changed_forbidden = build_phase4_evidence_claim_boundary(
        fact(safe_impact=["Improved reliability"])
    )
    assert first == second
    assert first.boundary_id != changed_allowed.boundary_id
    assert first.boundary_id != changed_forbidden.boundary_id


def test_technical_tag_order_and_input_order_do_not_change_results():
    first = fact(source_id="a", technical_tags=["Python", "FastAPI"])
    reordered = fact(source_id="a", technical_tags=["FastAPI", "Python"])
    assert build_phase4_evidence_claim_boundary(first) == build_phase4_evidence_claim_boundary(reordered)
    second = fact(source_id="b", mechanism="canonical serialization")
    forward = build_phase4_claim_boundaries_by_project([first, second])
    reverse = build_phase4_claim_boundaries_by_project([second, first])
    assert forward == reverse


def test_bounded_decision_sampling_and_truncation_are_deterministic():
    records = [
        fact(mechanism=f"deterministic mechanism {index}", source_id=f"src-{index}")
        for index in range(CLAIM_LIMITS["mechanism"] + 8)
    ]
    forward_boundary, forward_report = build_phase4_claim_boundaries_by_project(records)
    reverse_boundary, reverse_report = build_phase4_claim_boundaries_by_project(reversed(records))
    assert forward_boundary == reverse_boundary
    assert forward_report == reverse_report
    assert len(allowed(forward_boundary["workagent"], "mechanism")) == CLAIM_LIMITS["mechanism"]
    assert forward_report.truncated_claim_count >= 8
    assert len(forward_report.decisions) <= MAX_DECISION_SAMPLES


def test_machine_specific_absolute_prefix_does_not_change_boundary_id():
    first = fact(
        implementation=["C:/Users/alice/repo/backend/module.py::build_result"],
        evidence_fact_id="p4ef_same_safe_source",
    )
    second = fact(
        implementation=["D:/Users/bob/work/backend/module.py::build_result"],
        evidence_fact_id="p4ef_same_safe_source",
    )
    assert build_phase4_evidence_claim_boundary(first) == build_phase4_evidence_claim_boundary(second)


def test_same_fact_id_different_payload_is_conflicted_and_removed():
    first = fact(evidence_fact_id="p4ef_conflict", mechanism="deterministic ordering")
    second = fact(evidence_fact_id="p4ef_conflict", mechanism="canonical serialization")
    boundaries, report = build_phase4_claim_boundaries_by_project([first, second])
    assert boundaries == {}
    assert report.conflict_count == 1


def test_no_files_are_written(tmp_path):
    before = list(tmp_path.rglob("*"))
    build_phase4_claim_boundaries_by_project([fact()])
    assert list(tmp_path.rglob("*")) == before


def test_import_has_no_file_network_database_or_model_side_effects(monkeypatch):
    module_name = "backend.phase4_claim_boundary"
    before = set(sys.modules)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("unexpected read"))
    module = importlib.import_module(module_name)
    importlib.reload(module)
    newly_loaded = set(sys.modules) - before
    assert not any(name.startswith(("chromadb", "openai", "requests", "sqlite3")) for name in newly_loaded)


def test_step5_scored_facts_and_step7_zero_capabilities_pass_directly():
    inputs, _ = load_phase4_inputs()
    normalized, _ = dedupe_phase4_inputs(inputs)
    synthesized, _ = synthesize_phase4_evidence_facts(normalized)
    scored, _ = score_phase4_evidence_facts(synthesized)
    grouped, _ = extract_phase4_capabilities_by_project(scored)
    capabilities = [item for values in grouped.values() for item in values]
    boundaries, report = build_phase4_claim_boundaries_by_project(scored, capabilities)
    evidence_ids = {item.evidence_fact_id for item in scored}
    assert len(scored) == 283
    assert capabilities == []
    assert boundaries
    assert report.capability_claim_count == 0
    assert report.capability_fact_count == 0
    assert all(isinstance(value, Phase4ClaimBoundary) for value in boundaries.values())
    assert all(
        validate_phase4_claim_boundary(value, evidence_facts_by_id={item.evidence_fact_id: item for item in scored}).valid
        for value in boundaries.values()
    )
    support_text = "\n".join(note for value in boundaries.values() for note in value.notes)
    assert all(evidence_id in evidence_ids for evidence_id in evidence_ids if evidence_id in support_text)


def test_real_data_audit_is_read_only_and_weak_context_never_creates_allowed_claims():
    information = Path(__file__).resolve().parents[1] / "information"
    before = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    inputs, _ = load_phase4_inputs()
    normalized, _ = dedupe_phase4_inputs(inputs)
    synthesized, _ = synthesize_phase4_evidence_facts(normalized)
    scored, _ = score_phase4_evidence_facts(synthesized)
    weak = [item for item in scored if item.status is EvidenceStatus.WEAK]
    boundaries, report = build_phase4_claim_boundaries_by_project(scored)
    weak_boundaries = [build_phase4_evidence_claim_boundary(item) for item in weak]
    after = {path: path.stat().st_mtime_ns for path in information.rglob("*") if path.is_file()}
    assert len(weak) == 43
    assert all(value is None for value in weak_boundaries)
    assert report.weak_fact_blocked_count == 43
    assert report.project_boundaries_created == len(boundaries) == 11
    assert report.capability_claim_count == report.metric_claim_count == 0
    assert before == after


def test_project_subject_enum_is_backward_compatible_extension():
    assert ClaimSubjectType.EVIDENCE_FACT.value == "evidence_fact"
    assert ClaimSubjectType.CAPABILITY_FACT.value == "capability_fact"
    assert ClaimSubjectType.PROJECT.value == "project"
    assert set(GLOBAL_FORBIDDEN_POLICIES) <= set(build_phase4_evidence_claim_boundary(fact()).forbidden_claims)
