from __future__ import annotations

import ast
import builtins
from dataclasses import FrozenInstanceError
import inspect
import json
import os
from pathlib import Path

import pytest

import backend.project_evidence_coverage as coverage_module
from backend.project_evidence_coverage import (
    MAX_PRIORITIZED_FOLLOWUP_GAPS,
    CoverageCategory,
    CoverageDimensionResult,
    CoverageEvidenceRef,
    CoverageGap,
    CoverageReasonCode,
    CoverageReport,
    CoverageRequirement,
    CoverageState,
    GapPriority,
    GapPriorityReasonCode,
    build_project_evidence_coverage_report,
    prioritize_project_evidence_gaps,
)
from backend.project_evidence_models import (
    Confidence,
    EvidenceSourceRef,
    EvidenceStatus,
    EvidenceType,
    MetricSupport,
    ProjectEvidenceFact,
)


PROJECT_ID = "workagent"


def _ref(identifier: str, *, project_id: str = PROJECT_ID) -> CoverageEvidenceRef:
    return CoverageEvidenceRef(project_id=project_id, evidence_fact_id=identifier)


def _evidence_fact(
    evidence_fact_id: str,
    *,
    mechanism: str,
    implementation: tuple[str, ...],
    evidence_type: EvidenceType,
    technical_tags: tuple[str, ...] = (),
) -> ProjectEvidenceFact:
    return ProjectEvidenceFact(
        project_id=PROJECT_ID,
        evidence_fact_id=evidence_fact_id,
        mechanism=mechanism,
        implementation=list(implementation),
        evidence_type=evidence_type,
        technical_tags=list(technical_tags),
        status=EvidenceStatus.ACCEPTED,
        confidence=Confidence.HIGH,
        metric_support=MetricSupport.NONE,
        source_refs=[EvidenceSourceRef(
            source_type="project_change_evidence_card",
            source_id=f"src-{evidence_fact_id}",
            project_id=PROJECT_ID,
            content_hash="0" * 64,
        )],
    )


def _gap(
    category: CoverageCategory,
    *,
    state: CoverageState = CoverageState.MISSING,
    reason: CoverageReasonCode | None = None,
    refs: tuple[CoverageEvidenceRef, ...] = (),
    requirement_ids: tuple[str, ...] = (),
) -> CoverageGap:
    if reason is None:
        reason = (
            CoverageReasonCode.PARTIALLY_SUPPORTED
            if state is CoverageState.PARTIAL
            else CoverageReasonCode.CLAIM_BOUNDARY_RESTRICTED
            if state is CoverageState.BLOCKED
            else CoverageReasonCode.UNSUPPORTED
        )
    return CoverageGap(
        category=category,
        state=state,
        reason_code=reason,
        related_requirement_ids=requirement_ids,
        current_support_refs=refs,
    )


def _report(
    *gaps: CoverageGap,
    project_signal: bool = True,
    dimension_overrides: dict[CoverageCategory, CoverageDimensionResult] | None = None,
) -> CoverageReport:
    gap_by_category = {gap.category: gap for gap in gaps}
    overrides = dimension_overrides or {}
    dimensions: list[CoverageDimensionResult] = []
    for category in CoverageCategory:
        if category in overrides:
            dimensions.append(overrides[category])
            continue
        if category in gap_by_category:
            gap = gap_by_category[category]
            dimensions.append(CoverageDimensionResult(
                category=category,
                state=gap.state,
                reason_code=gap.reason_code,
                supporting_refs=tuple(
                    ref for ref in gap.current_support_refs if ref.project_id == PROJECT_ID
                ),
                requirement_ids=gap.related_requirement_ids,
            ))
            continue
        refs = (
            (_ref("pef-project-signal"),)
            if project_signal and category is CoverageCategory.PROJECT_IDENTITY
            else ()
        )
        dimensions.append(CoverageDimensionResult(
            category=category,
            state=CoverageState.COVERED,
            reason_code=CoverageReasonCode.SUPPORTED_BY_EVIDENCE,
            supporting_refs=refs,
        ))
    return CoverageReport(
        project_id=PROJECT_ID,
        dimensions=tuple(dimensions),
        gaps=tuple(gaps),
        overall_state=CoverageState.PARTIAL if gaps else CoverageState.COVERED,
    )


def _requirements(*items: tuple[str, tuple[str, ...]]) -> tuple[CoverageRequirement, ...]:
    return tuple(CoverageRequirement(requirement_id, targets) for requirement_id, targets in items)


def test_empty_gap_list_and_zero_capabilities_produce_no_followup_work():
    manual = _report()
    evaluated = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        capability_facts=(),
    )

    assert prioritize_project_evidence_gaps(coverage_report=manual) == ()
    assert prioritize_project_evidence_gaps(coverage_report=evaluated) == ()


@pytest.mark.parametrize("state", (CoverageState.COVERED, CoverageState.NOT_APPLICABLE))
def test_covered_or_not_applicable_dimension_cannot_be_overridden_by_a_gap(state):
    contradictory = _gap(CoverageCategory.VALIDATION_REPAIR)
    reason = (
        CoverageReasonCode.SUPPORTED_BY_EVIDENCE
        if state is CoverageState.COVERED
        else CoverageReasonCode.NO_RELEVANT_REQUIREMENT
    )
    report = _report(
        contradictory,
        dimension_overrides={
            contradictory.category: CoverageDimensionResult(
                category=contradictory.category,
                state=state,
                reason_code=reason,
            )
        },
    )

    assert prioritize_project_evidence_gaps(coverage_report=report) == ()


def test_jd_relevant_validation_and_retrieval_outrank_unrelated_missing_architecture():
    validation = _gap(
        CoverageCategory.VALIDATION_REPAIR,
        requirement_ids=("req-validation",),
    )
    retrieval = _gap(
        CoverageCategory.RETRIEVAL_RANKING,
        requirement_ids=("req-retrieval",),
    )
    architecture = _gap(CoverageCategory.ARCHITECTURE)
    requirements = _requirements(
        ("req-validation", ("schema validation and repair",)),
        ("req-retrieval", ("hybrid retrieval and reranking",)),
    )

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(architecture, validation, retrieval),
        jd_requirements=requirements,
    )

    assert tuple(item.category for item in prioritized[:2]) == (
        CoverageCategory.RETRIEVAL_RANKING,
        CoverageCategory.VALIDATION_REPAIR,
    )
    assert all(item.priority is GapPriority.HIGH for item in prioritized[:2])
    assert prioritized[2].category is CoverageCategory.ARCHITECTURE
    assert prioritized[2].priority is GapPriority.LOW


def test_partial_mechanism_support_outranks_speculative_missing_evidence():
    partial_ref = _ref("pef-partial-worker")
    implementation = _gap(
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        state=CoverageState.PARTIAL,
        refs=(partial_ref,),
    )
    architecture = _gap(CoverageCategory.ARCHITECTURE)

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(architecture, implementation),
    )

    assert tuple(item.category for item in prioritized) == (
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.ARCHITECTURE,
    )
    assert prioritized[0].priority is GapPriority.MEDIUM
    assert prioritized[0].reason_code is GapPriorityReasonCode.PARTIAL_SUPPORT_WORTH_EXPANDING
    assert prioritized[1].priority is GapPriority.LOW


def test_generic_llm_report_does_not_create_retrieval_or_reliability_work():
    generic_report = build_project_evidence_coverage_report(
        project_id=PROJECT_ID,
        evidence_facts=(_evidence_fact(
            "pef-generic-llm",
            mechanism="LLM response generation",
            implementation=("Called an LLM with a bounded prompt.",),
            evidence_type=EvidenceType.INTEGRATION,
            technical_tags=("llm", "prompt", "embedding"),
        ),),
    )

    prioritized = prioritize_project_evidence_gaps(coverage_report=generic_report)

    assert CoverageCategory.RETRIEVAL_RANKING not in {item.category for item in prioritized}
    assert CoverageCategory.RELIABILITY not in {item.category for item in prioritized}


def test_covered_metric_and_missing_project_identity_are_never_prioritized():
    covered_metric_gap = _gap(
        CoverageCategory.METRIC_IMPACT,
        reason=CoverageReasonCode.NO_METRIC_SUPPORT,
    )
    covered_metric_dimension = CoverageDimensionResult(
        category=CoverageCategory.METRIC_IMPACT,
        state=CoverageState.COVERED,
        reason_code=CoverageReasonCode.SUPPORTED_BY_EVIDENCE,
    )
    identity = _gap(CoverageCategory.PROJECT_IDENTITY)

    assert prioritize_project_evidence_gaps(
        coverage_report=_report(
            covered_metric_gap,
            dimension_overrides={CoverageCategory.METRIC_IMPACT: covered_metric_dimension},
        ),
    ) == ()
    assert prioritize_project_evidence_gaps(
        coverage_report=_report(identity, project_signal=False),
    ) == ()


def test_missing_metric_without_signal_is_not_actionable_but_partial_jd_metric_is_medium():
    missing = _gap(
        CoverageCategory.METRIC_IMPACT,
        reason=CoverageReasonCode.NO_METRIC_SUPPORT,
    )
    partial_ref = _ref("pef-approximate-latency")
    partial = _gap(
        CoverageCategory.METRIC_IMPACT,
        state=CoverageState.PARTIAL,
        refs=(partial_ref,),
        requirement_ids=("req-latency",),
    )
    metric_requirement = _requirements(("req-latency", ("measurable p95 latency",)))

    assert prioritize_project_evidence_gaps(coverage_report=_report(missing)) == ()
    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(partial),
        jd_requirements=metric_requirement,
    )

    assert len(prioritized) == 1
    assert prioritized[0].category is CoverageCategory.METRIC_IMPACT
    assert prioritized[0].priority is GapPriority.MEDIUM
    assert prioritized[0].searchable is True


def test_blocked_and_claim_boundary_restricted_gaps_fail_closed_even_with_jd_relevance():
    blocked = _gap(
        CoverageCategory.RETRIEVAL_RANKING,
        state=CoverageState.BLOCKED,
        refs=(_ref("pef-blocked"),),
        requirement_ids=("req-retrieval",),
    )

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(blocked),
        jd_requirements=_requirements(("req-retrieval", ("retrieval and ranking",))),
    )

    assert prioritized == ()


def test_wrong_project_support_reference_is_removed_and_cannot_create_partial_priority():
    foreign = _gap(
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        state=CoverageState.PARTIAL,
        refs=(_ref("pef-foreign", project_id="other-project"),),
    )
    unsupported = _gap(
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        state=CoverageState.PARTIAL,
    )

    foreign_result = prioritize_project_evidence_gaps(
        coverage_report=_report(foreign, project_signal=False),
    )
    unsupported_result = prioritize_project_evidence_gaps(
        coverage_report=_report(unsupported, project_signal=False),
    )

    assert foreign_result == unsupported_result == ()


def test_wrong_project_reason_cannot_be_overridden_by_jd_relevance():
    wrong_project = _gap(
        CoverageCategory.RETRIEVAL_RANKING,
        reason=CoverageReasonCode.WRONG_PROJECT,
        requirement_ids=("req-retrieval",),
    )

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(wrong_project),
        jd_requirements=_requirements(("req-retrieval", ("retrieval and ranking",))),
    )

    assert prioritized == ()


def test_followup_budget_is_hard_bounded_and_ignored_gaps_do_not_consume_it():
    actionable = tuple(
        _gap(category, requirement_ids=(f"req-{category.value}",))
        for category in (
            CoverageCategory.ARCHITECTURE,
            CoverageCategory.DATA_STORAGE,
            CoverageCategory.RETRIEVAL_RANKING,
            CoverageCategory.VALIDATION_REPAIR,
        )
    )
    blocked = _gap(CoverageCategory.OUTPUT_QUALITY, state=CoverageState.BLOCKED)
    requirements = tuple(
        CoverageRequirement(f"req-{gap.category.value}", (gap.category.value,))
        for gap in actionable
    )
    report = _report(blocked, *reversed(actionable))

    selected = prioritize_project_evidence_gaps(
        coverage_report=report,
        jd_requirements=requirements,
        max_followup_gaps=2,
    )

    assert len(selected) == 2
    assert all(item.priority is GapPriority.HIGH and item.searchable for item in selected)
    assert CoverageCategory.OUTPUT_QUALITY not in {item.category for item in selected}
    assert prioritize_project_evidence_gaps(
        coverage_report=report,
        jd_requirements=requirements,
        max_followup_gaps=0,
    ) == ()


def test_default_budget_never_selects_more_than_the_absolute_limit():
    categories = (
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.IMPLEMENTATION_MECHANISM,
        CoverageCategory.DATA_STORAGE,
        CoverageCategory.RETRIEVAL_RANKING,
        CoverageCategory.VALIDATION_REPAIR,
        CoverageCategory.OUTPUT_QUALITY,
    )
    gaps = tuple(
        _gap(category, requirement_ids=(f"req-{category.value}",))
        for category in categories
    )
    requirements = tuple(
        CoverageRequirement(f"req-{category.value}", (category.value,))
        for category in categories
    )

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(*gaps),
        jd_requirements=requirements,
    )

    assert len(prioritized) == MAX_PRIORITIZED_FOLLOWUP_GAPS


@pytest.mark.parametrize(
    ("budget", "error"),
    ((-1, ValueError), (MAX_PRIORITIZED_FOLLOWUP_GAPS + 1, ValueError), (True, TypeError), (1.5, TypeError)),
)
def test_invalid_or_expansive_followup_budget_fails_closed(budget, error):
    with pytest.raises(error):
        prioritize_project_evidence_gaps(
            coverage_report=_report(),
            max_followup_gaps=budget,
        )


def test_requirement_deduplication_and_input_order_are_deterministic():
    validation = _gap(CoverageCategory.VALIDATION_REPAIR)
    retrieval = _gap(CoverageCategory.RETRIEVAL_RANKING)
    forward_requirements = (
        CoverageRequirement("req-platform", ("validation",)),
        CoverageRequirement("req-platform", ("retrieval",)),
        CoverageRequirement("req-platform", ("validation",)),
    )

    forward = prioritize_project_evidence_gaps(
        coverage_report=_report(validation, retrieval),
        jd_requirements=forward_requirements,
    )
    reverse = prioritize_project_evidence_gaps(
        coverage_report=_report(retrieval, validation),
        jd_requirements=tuple(reversed(forward_requirements)),
    )

    assert forward == reverse
    assert tuple(item.category for item in forward) == (
        CoverageCategory.RETRIEVAL_RANKING,
        CoverageCategory.VALIDATION_REPAIR,
    )
    assert all(item.priority is GapPriority.LOW for item in forward)
    assert all(item.related_requirement_ids == () for item in forward)


def test_exact_duplicate_requirements_create_one_high_priority_gap():
    validation = _gap(CoverageCategory.VALIDATION_REPAIR)
    duplicate = CoverageRequirement("req-validation", ("validation and repair",))

    prioritized = prioritize_project_evidence_gaps(
        coverage_report=_report(validation),
        jd_requirements=(duplicate, duplicate, duplicate),
    )

    assert len(prioritized) == 1
    assert prioritized[0].priority is GapPriority.HIGH
    assert prioritized[0].related_requirement_ids == ("req-validation",)


def test_prioritizer_is_pure_immutable_and_contains_no_query_fields(monkeypatch):
    validation = _gap(
        CoverageCategory.VALIDATION_REPAIR,
        requirement_ids=("req-validation",),
    )
    report = _report(validation)
    before_report = json.dumps(report.to_dict(), sort_keys=True)
    before_environment = dict(os.environ)

    def _forbid_file_access(*args, **kwargs):
        raise AssertionError("gap prioritization must not access the filesystem")

    monkeypatch.setattr(builtins, "open", _forbid_file_access)
    result = prioritize_project_evidence_gaps(
        coverage_report=report,
        jd_requirements=_requirements(("req-validation", ("validation and repair",))),
    )

    assert json.dumps(report.to_dict(), sort_keys=True) == before_report
    assert dict(os.environ) == before_environment
    assert isinstance(result, tuple) and len(result) == 1
    assert not ({"query", "keywords", "symbols", "search_terms"} & set(result[0].to_dict()))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result[0].searchable = False  # type: ignore[misc]


def test_priority_module_has_no_retrieval_chroma_storage_or_environment_dependency():
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
        "backend.memory_store",
        "backend.chroma",
        "backend.project_retrieval",
        "backend.evidence_hybrid_retrieval",
        "backend.github_evidence_query_planner",
        "backend.project_query_planner",
        "backend.github_raw_storage",
    )

    assert not any(
        module_name.startswith(prefix)
        for module_name in imported_modules
        for prefix in forbidden_import_prefixes
    )
    assert "PersistentClient" not in source_text
    assert "os.environ" not in source_text
