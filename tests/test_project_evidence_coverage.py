from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError, asdict
import inspect
import json
import os
from pathlib import Path

import pytest

import backend.project_evidence_coverage as coverage_module
from backend.project_evidence_coverage import (
    MAX_SUPPORTING_REFS_PER_DIMENSION,
    CoverageCategory,
    CoverageRequirement,
    CoverageState,
    build_project_evidence_coverage_report,
)
from backend.project_evidence_models import (
    ClaimSubjectType,
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectCapabilityFact,
    ProjectClaimBoundary,
    ProjectEvidenceFact,
)


PROJECT_ID = "workagent"


def _source(source_id: str, *, project_id: str = PROJECT_ID) -> EvidenceSourceRef:
    digest = source_id.encode("utf-8").hex().ljust(64, "0")[:64]
    return EvidenceSourceRef(
        source_type="project_change_evidence_card",
        source_id=source_id,
        project_id=project_id,
        content_hash=digest,
    )


def _fact(
    evidence_fact_id: str,
    *,
    project_id: str = PROJECT_ID,
    mechanism: str = "Implemented a bounded deterministic worker queue",
    implementation: tuple[str, ...] = ("Added queue admission and worker scheduling.",),
    evidence_type: EvidenceType = EvidenceType.FEATURE,
    technical_tags: tuple[str, ...] = (),
    safe_impact: tuple[str, ...] = (),
    allowed_claims: tuple[str, ...] = (),
    metric_support: MetricSupport = MetricSupport.NONE,
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=project_id,
        mechanism=mechanism,
        implementation=list(implementation),
        source_refs=[_source(f"src-{evidence_fact_id}", project_id=project_id)],
        evidence_type=evidence_type,
        confidence=Confidence.HIGH,
        metric_support=metric_support,
        safe_impact=list(safe_impact),
        allowed_claims=list(allowed_claims),
        technical_tags=list(technical_tags),
        status=status,
        evidence_fact_id=evidence_fact_id,
    )


def _dimension(report, category: CoverageCategory):
    return next(item for item in report.dimensions if item.category is category)


def _boundary(
    evidence: ProjectEvidenceFact,
    *,
    metric_support: MetricSupport = MetricSupport.NONE,
    allowed_claims: tuple[str, ...] = (),
    forbidden_claims: tuple[str, ...] = (),
) -> ProjectClaimBoundary:
    return ProjectClaimBoundary(
        project_id=evidence.project_id,
        subject_type=ClaimSubjectType.EVIDENCE_FACT,
        subject_id=evidence.evidence_fact_id,
        metric_support=metric_support,
        allowed_claims=list(allowed_claims),
        forbidden_claims=list(forbidden_claims),
    )


def test_empty_input_never_fabricates_coverage_and_is_deterministic():
    first = build_project_evidence_coverage_report(project_id=PROJECT_ID)
    second = build_project_evidence_coverage_report(project_id=PROJECT_ID)

    assert first == second
    assert first.project_id == PROJECT_ID
    assert all(result.state is not CoverageState.COVERED for result in first.dimensions)
    assert _dimension(first, CoverageCategory.PROJECT_IDENTITY).state is CoverageState.MISSING
    assert _dimension(first, CoverageCategory.JD_MUST_HAVE).state is CoverageState.NOT_APPLICABLE
    assert tuple(gap.category for gap in first.gaps) == tuple(
        result.category
        for result in first.dimensions
        if result.state in {CoverageState.MISSING, CoverageState.PARTIAL, CoverageState.BLOCKED}
    )


def test_concrete_same_project_mechanism_is_covered_and_traceable():
    evidence = _fact("pef-mechanism")

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        capability_facts=(),
    )

    result = _dimension(report, CoverageCategory.IMPLEMENTATION_MECHANISM)
    assert result.state is CoverageState.COVERED
    assert [ref.evidence_fact_id for ref in result.supporting_refs] == [evidence.evidence_fact_id]
    assert _dimension(report, CoverageCategory.PROJECT_IDENTITY).state is CoverageState.COVERED


def test_architecture_requires_structured_specific_evidence():
    generic = _fact(
        "pef-generic",
        mechanism="Built an AI-powered application",
        implementation=("Built an AI-powered application",),
        evidence_type=EvidenceType.UNKNOWN,
    )
    concrete = _fact(
        "pef-architecture",
        mechanism="Separated ingestion, retrieval, and generation components",
        implementation=("Connected the ingestion workflow to an isolated retrieval service.",),
        evidence_type=EvidenceType.ARCHITECTURE,
    )

    generic_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(generic,),
    )
    concrete_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(concrete,),
    )

    assert _dimension(generic_report, CoverageCategory.ARCHITECTURE).state is not CoverageState.COVERED
    assert _dimension(concrete_report, CoverageCategory.ARCHITECTURE).state is CoverageState.COVERED


def test_generic_ai_prose_does_not_overcover_specialized_categories():
    evidence = _fact(
        "pef-generic-ai",
        mechanism="Built an AI-powered application",
        implementation=("Refined the prompt wording.",),
        evidence_type=EvidenceType.UNKNOWN,
        safe_impact=("Improved performance and quality.",),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    for category in (
        CoverageCategory.RETRIEVAL_RANKING,
        CoverageCategory.RELIABILITY,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
        CoverageCategory.METRIC_IMPACT,
    ):
        assert _dimension(report, category).state is not CoverageState.COVERED

    assert _dimension(report, CoverageCategory.IMPLEMENTATION_MECHANISM).state is not CoverageState.COVERED


@pytest.mark.parametrize(
    ("evidence_type", "mechanism", "implementation", "tags", "category"),
    [
        (
            EvidenceType.DATA_PERSISTENCE,
            "SQLite-backed project state",
            ("Persisted normalized project records in SQLite transactions.",),
            ("sqlite", "database", "persistence"),
            CoverageCategory.DATA_STORAGE,
        ),
        (
            EvidenceType.RETRIEVAL,
            "Hybrid retriever and deterministic reranker",
            ("Indexed evidence chunks and reranked search candidates.",),
            ("retrieval", "search", "reranking"),
            CoverageCategory.RETRIEVAL_RANKING,
        ),
        (
            EvidenceType.VALIDATION,
            "Schema validation with deterministic repair",
            ("Validated structured output and repaired invalid fields before retry.",),
            ("validation", "repair", "retry"),
            CoverageCategory.VALIDATION_REPAIR,
        ),
        (
            EvidenceType.VALIDATION,
            "Output quality gate",
            ("Rejected unsupported output claims before final generation.",),
            ("output_quality", "claim_validation"),
            CoverageCategory.OUTPUT_QUALITY,
        ),
    ],
)
def test_explicit_structured_mechanisms_cover_only_their_supported_category(
    evidence_type,
    mechanism,
    implementation,
    tags,
    category,
):
    evidence = _fact(
        f"pef-{category.value}",
        mechanism=mechanism,
        implementation=implementation,
        evidence_type=evidence_type,
        technical_tags=tags,
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    result = _dimension(report, category)
    assert result.state is CoverageState.COVERED
    assert result.supporting_refs[0].evidence_fact_id == evidence.evidence_fact_id


def test_llm_use_does_not_imply_retrieval_or_ranking():
    evidence = _fact(
        "pef-llm",
        mechanism="LLM response generation",
        implementation=("Called an LLM with a bounded prompt.",),
        evidence_type=EvidenceType.INTEGRATION,
        technical_tags=("llm", "prompt"),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    assert _dimension(report, CoverageCategory.RETRIEVAL_RANKING).state is not CoverageState.COVERED


def test_embedding_generation_alone_does_not_imply_retrieval_or_ranking():
    evidence = _fact(
        "pef-embedding",
        mechanism="Embedding generation",
        implementation=("Generated bounded vector representations for text.",),
        evidence_type=EvidenceType.INTEGRATION,
        technical_tags=("embedding", "vector"),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    assert _dimension(report, CoverageCategory.RETRIEVAL_RANKING).state is not CoverageState.COVERED


def test_reliability_requires_an_explicit_control_not_prompt_wording():
    prompt_only = _fact(
        "pef-prompt",
        mechanism="Careful prompt wording",
        implementation=("Edited the system prompt to request accurate answers.",),
        evidence_type=EvidenceType.CONFIGURATION,
        technical_tags=("prompt",),
    )
    grounding = _fact(
        "pef-grounding",
        mechanism="Evidence grounding and unsupported-claim fallback",
        implementation=("Blocked unsupported claims and used a safe fallback below the confidence threshold.",),
        evidence_type=EvidenceType.FAILURE_RECOVERY,
        technical_tags=("grounding", "fallback", "claim_validation"),
    )

    prompt_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(prompt_only,),
    )
    grounding_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(grounding,),
    )

    assert _dimension(prompt_report, CoverageCategory.RELIABILITY).state is not CoverageState.COVERED
    assert _dimension(grounding_report, CoverageCategory.RELIABILITY).state is CoverageState.COVERED


@pytest.mark.parametrize("prose", ["improved", "optimized", "faster", "reduced latency"])
def test_unsupported_metric_prose_never_creates_metric_coverage(prose):
    evidence = _fact(
        f"pef-{prose.replace(' ', '-')}",
        mechanism="Performance tuning",
        implementation=("Adjusted worker scheduling.",),
        safe_impact=(prose,),
        metric_support=MetricSupport.NONE,
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    assert _dimension(report, CoverageCategory.METRIC_IMPACT).state is not CoverageState.COVERED


@pytest.mark.parametrize(
    "status",
    (EvidenceStatus.SUPPORTING, EvidenceStatus.WEAK, EvidenceStatus.REJECTED),
)
def test_nonaccepted_evidence_cannot_create_affirmative_coverage(status):
    evidence = _fact(
        f"pef-{status.value}",
        evidence_type=EvidenceType.RETRIEVAL,
        mechanism="Vector retrieval and deterministic reranking",
        implementation=("Retrieved candidates and applied a bounded reranker.",),
        technical_tags=("retrieval", "reranking"),
        status=status,
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    assert _dimension(report, CoverageCategory.RETRIEVAL_RANKING).state is not CoverageState.COVERED
    assert _dimension(report, CoverageCategory.IMPLEMENTATION_MECHANISM).state is not CoverageState.COVERED


def test_explicit_and_approximate_metric_support_remain_distinct():
    explicit = _fact(
        "pef-explicit-metric",
        safe_impact=("Reduced p95 latency by 30%.",),
        allowed_claims=("Reduced p95 latency by 30%.",),
        metric_support=MetricSupport.EXPLICIT,
    )
    approximate = _fact(
        "pef-approximate-metric",
        safe_impact=("Reduced latency by approximately 30%.",),
        allowed_claims=("Reduced latency by approximately 30%.",),
        metric_support=MetricSupport.APPROXIMATE,
    )

    explicit_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(explicit,),
    )
    approximate_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(approximate,),
    )

    assert _dimension(explicit_report, CoverageCategory.METRIC_IMPACT).state is CoverageState.COVERED
    assert _dimension(approximate_report, CoverageCategory.METRIC_IMPACT).state is CoverageState.PARTIAL


@pytest.mark.parametrize(
    ("support", "impact"),
    [
        (MetricSupport.EXPLICIT, "Improved performance."),
        (MetricSupport.APPROXIMATE, "Reduced latency by 30%."),
    ],
)
def test_metric_support_label_without_a_policy_valid_metric_does_not_cover(support, impact):
    evidence = _fact(
        "pef-invalid-metric",
        safe_impact=(impact,),
        metric_support=support,
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
    )

    assert _dimension(report, CoverageCategory.METRIC_IMPACT).state is not CoverageState.COVERED


def test_matching_claim_boundary_can_restrict_metric_coverage():
    evidence = _fact(
        "pef-boundary-metric",
        safe_impact=("Reduced latency by 30%.",),
        allowed_claims=("Reduced latency by 30%.",),
        metric_support=MetricSupport.EXPLICIT,
    )
    boundary = _boundary(
        evidence,
        metric_support=MetricSupport.NONE,
        forbidden_claims=("metric:Reduced latency by 30%.",),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        claim_boundaries=(boundary,),
    )

    result = _dimension(report, CoverageCategory.METRIC_IMPACT)
    assert result.state is CoverageState.BLOCKED
    assert result.reason_code == "claim_boundary_restricted"
    assert any(ref.claim_boundary_id == boundary.boundary_id for ref in result.supporting_refs)


def test_allowed_claim_boundary_support_remains_traceable():
    claim = "Reduced p95 latency by 30%."
    evidence = _fact(
        "pef-allowed-metric",
        safe_impact=(claim,),
        allowed_claims=(claim,),
        metric_support=MetricSupport.EXPLICIT,
    )
    boundary = _boundary(
        evidence,
        metric_support=MetricSupport.EXPLICIT,
        allowed_claims=(f"metric:{claim}",),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        claim_boundaries=(boundary,),
    )

    result = _dimension(report, CoverageCategory.METRIC_IMPACT)
    assert result.state is CoverageState.COVERED
    assert any(ref.evidence_fact_id == evidence.evidence_fact_id for ref in result.supporting_refs)
    assert any(ref.claim_boundary_id == boundary.boundary_id for ref in result.supporting_refs)


def test_nonmetric_claim_boundary_can_block_or_allow_the_exact_dimension():
    evidence = _fact(
        "pef-retrieval-boundary",
        mechanism="Hybrid retrieval and deterministic reranking",
        implementation=("Indexed evidence chunks and reranked search candidates.",),
        evidence_type=EvidenceType.RETRIEVAL,
        technical_tags=("retrieval", "reranking"),
    )
    blocked = _boundary(
        evidence,
        forbidden_claims=("retrieval:unsupported retrieval claim",),
    )
    allowed = _boundary(
        evidence,
        allowed_claims=("retrieval:hybrid retrieval and deterministic reranking",),
    )

    blocked_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        claim_boundaries=(blocked,),
    )
    allowed_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        claim_boundaries=(allowed,),
    )

    assert _dimension(blocked_report, CoverageCategory.RETRIEVAL_RANKING).state is CoverageState.BLOCKED
    assert _dimension(allowed_report, CoverageCategory.RETRIEVAL_RANKING).state is CoverageState.COVERED


def test_wrong_project_evidence_capabilities_and_boundaries_are_fail_closed():
    foreign = _fact(
        "pef-foreign",
        project_id="other-project",
        evidence_type=EvidenceType.RETRIEVAL,
        mechanism="Vector search and reranking",
        implementation=("Indexed and reranked repository chunks.",),
        technical_tags=("retrieval", "reranking"),
        metric_support=MetricSupport.EXPLICIT,
        safe_impact=("Improved recall by 20%.",),
        allowed_claims=("Improved recall by 20%.",),
    )
    foreign_capability = ProjectCapabilityFact(
        project_id="other-project",
        capability_type="retrieval_ranking",
        present=True,
        source_evidence_fact_ids=[foreign.evidence_fact_id],
        confidence=Confidence.HIGH,
        mechanisms=["Vector retrieval"],
        metric_support=MetricSupport.EXPLICIT,
    )
    foreign_boundary = _boundary(
        foreign,
        metric_support=MetricSupport.EXPLICIT,
        allowed_claims=("Improved recall by 20%.",),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(foreign,),
        capability_facts=(foreign_capability,),
        claim_boundaries=(foreign_boundary,),
    )

    assert _dimension(report, CoverageCategory.PROJECT_IDENTITY).state is CoverageState.MISSING
    assert all(result.state is not CoverageState.COVERED for result in report.dimensions)
    assert not any(result.supporting_refs for result in report.dimensions)


def test_zero_capability_facts_is_a_valid_positive_evidence_state():
    evidence = _fact("pef-no-capability")

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        capability_facts=(),
    )

    assert _dimension(report, CoverageCategory.IMPLEMENTATION_MECHANISM).state is CoverageState.COVERED


def test_present_capability_requires_resolved_same_project_evidence_before_it_contributes():
    evidence = _fact("pef-capability-source")
    capability = ProjectCapabilityFact(
        project_id=PROJECT_ID,
        capability_type="retrieval_and_reranking",
        present=True,
        source_evidence_fact_ids=[evidence.evidence_fact_id],
        mechanisms=["Deterministic reranking"],
    )

    resolved = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(evidence,),
        capability_facts=(capability,),
    )
    unresolved = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        capability_facts=(capability,),
    )

    assert _dimension(resolved, CoverageCategory.RETRIEVAL_RANKING).state is CoverageState.COVERED
    assert any(
        ref.capability_fact_id == capability.capability_id
        for ref in _dimension(resolved, CoverageCategory.RETRIEVAL_RANKING).supporting_refs
    )
    assert _dimension(unresolved, CoverageCategory.RETRIEVAL_RANKING).state is not CoverageState.COVERED


@pytest.mark.parametrize("project_id", ["", "project with spaces", "../project", "project/id"])
def test_requested_project_identity_uses_the_authoritative_normalizer(project_id):
    with pytest.raises(ValueError, match="normalized project identifier"):
        build_project_evidence_coverage_report(project_id=project_id)


def test_normalized_jd_requirements_are_stable_and_unmatched_requirements_remain_gaps():
    evidence = _fact(
        "pef-sqlite",
        mechanism="SQLite persistence",
        implementation=("Persisted records in SQLite.",),
        evidence_type=EvidenceType.DATA_PERSISTENCE,
        technical_tags=("sqlite", "persistence"),
    )
    requirements = (
        CoverageRequirement(requirement_id="req-python", target_terms=("python",)),
        CoverageRequirement(requirement_id="req-sqlite", target_terms=("sqlite",)),
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        jd_requirements=requirements,
        evidence_facts=(evidence,),
    )

    result = _dimension(report, CoverageCategory.JD_MUST_HAVE)
    assert result.state is CoverageState.PARTIAL
    assert result.requirement_ids == ("req-python", "req-sqlite")
    jd_gap = next(gap for gap in report.gaps if gap.category is CoverageCategory.JD_MUST_HAVE)
    assert jd_gap.related_requirement_ids == ("req-python",)
    assert any(ref.evidence_fact_id == evidence.evidence_fact_id for ref in result.supporting_refs)


def test_input_order_does_not_change_report_or_reference_order():
    first = _fact("pef-z", technical_tags=("sqlite",), evidence_type=EvidenceType.DATA_PERSISTENCE)
    second = _fact("pef-a", technical_tags=("sqlite",), evidence_type=EvidenceType.DATA_PERSISTENCE)
    first_boundary = _boundary(first, allowed_claims=("Implemented SQLite persistence.",))
    second_boundary = _boundary(second, allowed_claims=("Implemented SQLite persistence.",))

    forward = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(first, second),
        claim_boundaries=(first_boundary, second_boundary),
    )
    reverse = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(second, first),
        claim_boundaries=(second_boundary, first_boundary),
    )

    assert forward == reverse
    refs = _dimension(forward, CoverageCategory.DATA_STORAGE).supporting_refs
    assert tuple(ref.evidence_fact_id for ref in refs) == tuple(
        sorted(ref.evidence_fact_id for ref in refs if ref.evidence_fact_id)
    )


def test_supporting_references_are_bounded_and_contain_identifiers_only():
    evidence = tuple(
        _fact(
            f"pef-storage-{index:02d}",
            evidence_type=EvidenceType.DATA_PERSISTENCE,
            mechanism="SQLite persistence",
            implementation=("Persisted normalized records in SQLite.",),
            technical_tags=("sqlite", "persistence"),
        )
        for index in range(MAX_SUPPORTING_REFS_PER_DIMENSION + 5)
    )

    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=reversed(evidence),
    )

    refs = _dimension(report, CoverageCategory.DATA_STORAGE).supporting_refs
    assert len(refs) == MAX_SUPPORTING_REFS_PER_DIMENSION
    assert tuple(ref.evidence_fact_id for ref in refs) == tuple(sorted(ref.evidence_fact_id for ref in refs))
    for ref in refs:
        payload = asdict(ref)
        assert set(payload) == {
            "project_id",
            "evidence_fact_id",
            "capability_fact_id",
            "claim_boundary_id",
            "chunk_id",
            "source_id",
        }
        serialized = json.dumps(payload, sort_keys=True)
        assert "Persisted normalized records" not in serialized
        assert "raw_text" not in serialized
        assert "raw_patch" not in serialized


def test_models_are_immutable_and_evaluation_has_no_input_or_environment_side_effects(monkeypatch):
    evidence = _fact("pef-pure")
    before_fact = evidence.to_dict()
    before_environment = dict(os.environ)

    def _forbid_file_access(*args, **kwargs):
        raise AssertionError("coverage evaluation must not access the filesystem")

    monkeypatch.setattr(builtins, "open", _forbid_file_access)
    report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=[evidence],
        capability_facts=[],
        claim_boundaries=[],
    )

    assert evidence.to_dict() == before_fact
    assert dict(os.environ) == before_environment
    assert isinstance(report.dimensions, tuple)
    assert isinstance(report.gaps, tuple)
    assert all(isinstance(result.supporting_refs, tuple) for result in report.dimensions)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        report.project_id = "mutated"  # type: ignore[misc]


def test_coverage_module_has_no_retrieval_chroma_or_raw_storage_dependency():
    source_path = Path(inspect.getsourcefile(coverage_module) or "")
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    forbidden_import_prefixes = (
        "chromadb",
        "backend.chroma",
        "backend.project_retrieval",
        "backend.evidence_hybrid_retrieval",
        "backend.github_raw_storage",
    )
    assert not any(
        module_name.startswith(prefix)
        for module_name in imported_modules
        for prefix in forbidden_import_prefixes
    )
    assert "PersistentClient" not in source_text
