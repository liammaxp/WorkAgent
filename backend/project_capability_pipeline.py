"""Authoritative orchestration for the Project Capability Memory lifecycle.

This module connects the existing grouping, scoring, claim-policy, fact-builder,
and memory-persistence layers.  It contains no lifecycle algorithms, production
integration, command-line entry point, network access, or default artifact write.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend import evidence_memory as evidence_memory_module
from backend import project_capability_boundaries as boundary_module
from backend import project_capability_builder as builder_module
from backend import project_capability_grouping as grouping_module
from backend import project_capability_memory as capability_memory_module
from backend import project_capability_scoring as scoring_module
from backend import project_evidence_memory as evidence_memory_store
from backend.project_capability_memory import (
    PROJECT_CAPABILITY_MEMORY_PATH,
    ProjectCapabilityMemory,
    ProjectCapabilitySourceArtifact,
)
from backend.project_evidence_memory import ProjectEvidenceMemorySnapshot
from backend.project_evidence_models import ProjectCapabilityFact, ProjectEvidenceMemory


PROJECT_CAPABILITY_PIPELINE_STATUSES = frozenset({
    "ready",
    "empty",
    "source_missing",
    "source_invalid",
    "failed",
})
PROJECT_CAPABILITY_PROJECT_PIPELINE_STATUSES = frozenset({"ready", "empty", "failed"})
_SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SAFE_PATH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _codes(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(not isinstance(value, str) or not _SAFE_CODE_RE.fullmatch(value) for value in normalized):
        raise ValueError("pipeline warnings and errors must be safe codes")
    return normalized


def _counts(values: Mapping[str, int]) -> Mapping[str, int]:
    normalized: dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _SAFE_CODE_RE.fullmatch(key):
            raise ValueError("pipeline count keys must be safe codes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("pipeline counts must be non-negative integers")
        normalized[key] = value
    return MappingProxyType({key: normalized[key] for key in sorted(normalized)})


def _merge_count_mappings(
    values: Sequence[Mapping[str, int]],
) -> Mapping[str, int]:
    merged: Counter[str] = Counter()
    for value in values:
        merged.update(value)
    return merged


@dataclass(frozen=True)
class ProjectCapabilityProjectPipelineResult:
    project_id: str
    status: str
    evidence_fact_count: int
    claim_boundary_count: int
    candidate_count: int
    matched_evidence_count: int
    unmatched_evidence_count: int
    ambiguous_evidence_count: int
    skipped_evidence_count: int
    assessment_count: int
    eligible_assessment_count: int
    policy_count: int
    eligible_policy_count: int
    build_result_count: int
    built_fact_count: int
    capability_fact_ids: tuple[str, ...]
    assessment_status_counts: Mapping[str, int]
    policy_status_counts: Mapping[str, int]
    build_status_counts: Mapping[str, int]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROJECT_CAPABILITY_PROJECT_PIPELINE_STATUSES:
            raise ValueError("unsupported project pipeline status")
        for name in (
            "evidence_fact_count", "claim_boundary_count", "candidate_count",
            "matched_evidence_count", "unmatched_evidence_count",
            "ambiguous_evidence_count", "skipped_evidence_count", "assessment_count",
            "eligible_assessment_count", "policy_count", "eligible_policy_count",
            "build_result_count", "built_fact_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "capability_fact_ids", tuple(sorted(set(self.capability_fact_ids))))
        object.__setattr__(self, "assessment_status_counts", _counts(self.assessment_status_counts))
        object.__setattr__(self, "policy_status_counts", _counts(self.policy_status_counts))
        object.__setattr__(self, "build_status_counts", _counts(self.build_status_counts))
        object.__setattr__(self, "warnings", _codes(self.warnings))
        object.__setattr__(self, "errors", _codes(self.errors))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "status": self.status,
            "evidence_fact_count": self.evidence_fact_count,
            "claim_boundary_count": self.claim_boundary_count,
            "candidate_count": self.candidate_count,
            "matched_evidence_count": self.matched_evidence_count,
            "unmatched_evidence_count": self.unmatched_evidence_count,
            "ambiguous_evidence_count": self.ambiguous_evidence_count,
            "skipped_evidence_count": self.skipped_evidence_count,
            "assessment_count": self.assessment_count,
            "eligible_assessment_count": self.eligible_assessment_count,
            "policy_count": self.policy_count,
            "eligible_policy_count": self.eligible_policy_count,
            "build_result_count": self.build_result_count,
            "built_fact_count": self.built_fact_count,
            "capability_fact_ids": list(self.capability_fact_ids),
            "assessment_status_counts": dict(self.assessment_status_counts),
            "policy_status_counts": dict(self.policy_status_counts),
            "build_status_counts": dict(self.build_status_counts),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ProjectCapabilityPipelineResult:
    status: str
    source_load_status: str
    source_schema_version: str | None
    source_content_hash: str | None
    source_file_sha256: str | None
    project_results: tuple[ProjectCapabilityProjectPipelineResult, ...]
    source_project_count: int
    source_evidence_fact_count: int
    source_claim_boundary_count: int
    source_capability_fact_count: int
    candidate_count: int
    matched_evidence_count: int
    unmatched_evidence_count: int
    ambiguous_evidence_count: int
    skipped_evidence_count: int
    assessment_count: int
    eligible_assessment_count: int
    assessment_status_counts: Mapping[str, int]
    policy_count: int
    eligible_policy_count: int
    policy_status_counts: Mapping[str, int]
    build_result_count: int
    build_status_counts: Mapping[str, int]
    capability_fact_count: int
    memory: ProjectCapabilityMemory | None
    persisted_path: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in PROJECT_CAPABILITY_PIPELINE_STATUSES:
            raise ValueError("unsupported pipeline status")
        for name in (
            "source_project_count", "source_evidence_fact_count",
            "source_claim_boundary_count", "source_capability_fact_count",
            "candidate_count", "matched_evidence_count", "unmatched_evidence_count",
            "ambiguous_evidence_count", "skipped_evidence_count", "assessment_count",
            "eligible_assessment_count", "policy_count", "eligible_policy_count",
            "build_result_count", "capability_fact_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "project_results", tuple(self.project_results))
        object.__setattr__(self, "assessment_status_counts", _counts(self.assessment_status_counts))
        object.__setattr__(self, "policy_status_counts", _counts(self.policy_status_counts))
        object.__setattr__(self, "build_status_counts", _counts(self.build_status_counts))
        object.__setattr__(self, "warnings", _codes(self.warnings))
        object.__setattr__(self, "errors", _codes(self.errors))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_load_status": self.source_load_status,
            "source_schema_version": self.source_schema_version,
            "source_content_hash": self.source_content_hash,
            "source_file_sha256": self.source_file_sha256,
            "project_results": [item.to_safe_dict() for item in self.project_results],
            "source_project_count": self.source_project_count,
            "source_evidence_fact_count": self.source_evidence_fact_count,
            "source_claim_boundary_count": self.source_claim_boundary_count,
            "source_capability_fact_count": self.source_capability_fact_count,
            "candidate_count": self.candidate_count,
            "matched_evidence_count": self.matched_evidence_count,
            "unmatched_evidence_count": self.unmatched_evidence_count,
            "ambiguous_evidence_count": self.ambiguous_evidence_count,
            "skipped_evidence_count": self.skipped_evidence_count,
            "assessment_count": self.assessment_count,
            "eligible_assessment_count": self.eligible_assessment_count,
            "assessment_status_counts": dict(self.assessment_status_counts),
            "policy_count": self.policy_count,
            "eligible_policy_count": self.eligible_policy_count,
            "policy_status_counts": dict(self.policy_status_counts),
            "build_result_count": self.build_result_count,
            "build_status_counts": dict(self.build_status_counts),
            "capability_fact_count": self.capability_fact_count,
            "memory": self.memory.to_safe_dict() if self.memory is not None else None,
            "persisted_path": self.persisted_path,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": _thaw(self.diagnostics),
        }


class _PipelineInvariantError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _ProjectExecution:
    result: ProjectCapabilityProjectPipelineResult
    facts: tuple[ProjectCapabilityFact, ...]


def _project_result(
    *,
    project: ProjectEvidenceMemory,
    status: str,
    grouping: Any = None,
    assessments: Sequence[Any] = (),
    policies: Sequence[Any] = (),
    build_results: Sequence[Any] = (),
    facts: Sequence[ProjectCapabilityFact] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> ProjectCapabilityProjectPipelineResult:
    return ProjectCapabilityProjectPipelineResult(
        project_id=project.project_id,
        status=status,
        evidence_fact_count=len(project.evidence_facts),
        claim_boundary_count=len(project.claim_boundaries),
        candidate_count=len(grouping.candidates) if grouping is not None else 0,
        matched_evidence_count=len(grouping.matched_evidence_ids) if grouping is not None else 0,
        unmatched_evidence_count=len(grouping.unmatched_evidence_ids) if grouping is not None else 0,
        ambiguous_evidence_count=len(grouping.ambiguous_evidence_ids) if grouping is not None else 0,
        skipped_evidence_count=len(grouping.skipped_evidence_ids) if grouping is not None else 0,
        assessment_count=len(assessments),
        eligible_assessment_count=sum(item.eligibility_status == "eligible" for item in assessments),
        policy_count=len(policies),
        eligible_policy_count=sum(item.policy_status == "eligible" for item in policies),
        build_result_count=len(build_results),
        built_fact_count=sum(item.build_status == "built" for item in build_results),
        capability_fact_ids=tuple(fact.capability_id for fact in facts),
        assessment_status_counts=Counter(item.eligibility_status for item in assessments),
        policy_status_counts=Counter(item.policy_status for item in policies),
        build_status_counts=Counter(item.build_status for item in build_results),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _validate_grouping(project: ProjectEvidenceMemory, grouping: Any) -> None:
    expected_ids = {fact.evidence_fact_id for fact in project.evidence_facts}
    if grouping.project_id != project.project_id:
        raise _PipelineInvariantError("cross_project_lifecycle_record")
    if set(grouping.input_evidence_ids) != expected_ids or len(grouping.input_evidence_ids) != len(expected_ids):
        raise _PipelineInvariantError("project_grouping_accounting_mismatch")
    categories = (
        set(grouping.matched_evidence_ids),
        set(grouping.unmatched_evidence_ids),
        set(grouping.ambiguous_evidence_ids),
        set(grouping.skipped_evidence_ids),
    )
    if any(left & right for index, left in enumerate(categories) for right in categories[index + 1:]):
        raise _PipelineInvariantError("project_grouping_accounting_mismatch")
    if set().union(*categories) != expected_ids or sum(map(len, categories)) != len(expected_ids):
        raise _PipelineInvariantError("project_grouping_accounting_mismatch")
    seen_types: set[str] = set()
    for candidate in grouping.candidates:
        if candidate.project_id != project.project_id:
            raise _PipelineInvariantError("cross_project_lifecycle_record")
        if candidate.capability_type in seen_types:
            raise _PipelineInvariantError("duplicate_candidate_identity")
        if not set(candidate.supporting_evidence_ids) <= expected_ids:
            raise _PipelineInvariantError("candidate_evidence_mismatch")
        seen_types.add(candidate.capability_type)


def _validate_assessments(project_id: str, candidates: Sequence[Any], assessments: Sequence[Any]) -> None:
    candidates_by_type = {item.capability_type: item for item in candidates}
    if len(candidates_by_type) != len(candidates) or len(assessments) != len(candidates):
        raise _PipelineInvariantError("candidate_assessment_mismatch")
    assessments_by_type = {item.capability_type: item for item in assessments}
    if len(assessments_by_type) != len(assessments) or set(assessments_by_type) != set(candidates_by_type):
        raise _PipelineInvariantError("candidate_assessment_mismatch")
    for capability_type, assessment in assessments_by_type.items():
        candidate = candidates_by_type[capability_type]
        if assessment.project_id != project_id:
            raise _PipelineInvariantError("cross_project_lifecycle_record")
        if assessment.supporting_evidence_ids != candidate.supporting_evidence_ids:
            raise _PipelineInvariantError("candidate_assessment_mismatch")


def _validate_policies(
    project_id: str,
    candidates: Sequence[Any],
    assessments: Sequence[Any],
    policies: Sequence[Any],
) -> None:
    candidates_by_type = {item.capability_type: item for item in candidates}
    eligible_types = {
        item.capability_type for item in assessments if item.eligibility_status == "eligible"
    }
    policies_by_type = {item.capability_type: item for item in policies}
    if len(policies_by_type) != len(policies) or set(policies_by_type) != eligible_types:
        raise _PipelineInvariantError("eligible_assessment_missing_policy")
    for capability_type, policy in policies_by_type.items():
        if policy.project_id != project_id:
            raise _PipelineInvariantError("cross_project_lifecycle_record")
        if policy.supporting_evidence_ids != candidates_by_type[capability_type].supporting_evidence_ids:
            raise _PipelineInvariantError("policy_evidence_mismatch")


def _validate_build_results(
    project_id: str,
    candidates: Sequence[Any],
    policies: Sequence[Any],
    build_results: Sequence[Any],
) -> tuple[ProjectCapabilityFact, ...]:
    candidates_by_type = {item.capability_type: item for item in candidates}
    eligible_types = {item.capability_type for item in policies if item.policy_status == "eligible"}
    results_by_type = {item.capability_type: item for item in build_results}
    if len(results_by_type) != len(build_results) or set(results_by_type) != eligible_types:
        raise _PipelineInvariantError("eligible_policy_missing_build_result")
    facts: list[ProjectCapabilityFact] = []
    for capability_type in sorted(results_by_type):
        result = results_by_type[capability_type]
        if result.project_id != project_id:
            raise _PipelineInvariantError("cross_project_lifecycle_record")
        if result.build_status != "built" or result.fact is None:
            raise _PipelineInvariantError("eligible_policy_build_failed")
        fact = result.fact
        candidate = candidates_by_type[capability_type]
        if (
            fact.project_id != project_id
            or fact.capability_type != capability_type
            or tuple(fact.source_evidence_fact_ids) != candidate.supporting_evidence_ids
            or fact.present is not True
        ):
            raise _PipelineInvariantError("cross_project_lifecycle_record")
        facts.append(fact)
    fact_ids = [fact.capability_id for fact in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise _PipelineInvariantError("duplicate_final_capability_identity")
    return tuple(facts)


def _process_project(project: ProjectEvidenceMemory) -> _ProjectExecution:
    grouping = None
    assessments: tuple[Any, ...] = ()
    policies: tuple[Any, ...] = ()
    build_results: tuple[Any, ...] = ()
    facts: tuple[ProjectCapabilityFact, ...] = ()
    warnings: set[str] = set()
    errors: set[str] = set()

    if any(fact.project_id != project.project_id for fact in project.evidence_facts) or any(
        boundary.project_id != project.project_id for boundary in project.claim_boundaries
    ):
        errors.add("cross_project_lifecycle_record")
    else:
        try:
            grouping = grouping_module.group_project_evidence_facts(
                project_id=project.project_id,
                evidence_facts=tuple(project.evidence_facts),
            )
            _validate_grouping(project, grouping)
        except _PipelineInvariantError as exc:
            errors.add(exc.code)
        except Exception:
            errors.add("project_grouping_failed")

    if not errors:
        if not grouping.candidates:
            warnings.add("project_has_no_capability_candidates")
        try:
            assessments = scoring_module.assess_project_capability_candidates(
                project_id=project.project_id,
                candidates=grouping.candidates,
                evidence_facts=tuple(project.evidence_facts),
            )
            _validate_assessments(project.project_id, grouping.candidates, assessments)
        except _PipelineInvariantError as exc:
            errors.add(exc.code)
        except Exception:
            errors.add("candidate_assessment_mismatch")

    if not errors:
        try:
            policies = boundary_module.inherit_project_capability_claim_policies(
                project_id=project.project_id,
                candidates=grouping.candidates,
                assessments=assessments,
                evidence_facts=tuple(project.evidence_facts),
                claim_boundaries=tuple(project.claim_boundaries),
            )
            _validate_policies(project.project_id, grouping.candidates, assessments, policies)
        except _PipelineInvariantError as exc:
            errors.add(exc.code)
        except Exception:
            errors.add("eligible_assessment_missing_policy")

    if not errors:
        candidates_by_type = {item.capability_type: item for item in grouping.candidates}
        assessments_by_type = {item.capability_type: item for item in assessments}
        eligible_policies = tuple(item for item in policies if item.policy_status == "eligible")
        eligible_types = tuple(sorted(item.capability_type for item in eligible_policies))
        try:
            build_results = builder_module.build_project_capability_facts(
                project_id=project.project_id,
                candidates=tuple(candidates_by_type[item] for item in eligible_types),
                assessments=tuple(assessments_by_type[item] for item in eligible_types),
                policies=eligible_policies,
                evidence_facts=tuple(project.evidence_facts),
            )
            facts = _validate_build_results(
                project.project_id, grouping.candidates, policies, build_results
            )
        except _PipelineInvariantError as exc:
            errors.add(exc.code)
        except Exception:
            errors.add("eligible_policy_missing_build_result")

    if not facts:
        warnings.add("project_has_no_verified_capabilities")
    status = "failed" if errors else ("ready" if facts else "empty")
    result = _project_result(
        project=project,
        status=status,
        grouping=grouping,
        assessments=assessments,
        policies=policies,
        build_results=build_results,
        facts=facts,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
    return _ProjectExecution(result=result, facts=facts if not errors else ())


def _source_failure_result(
    *,
    status: str,
    source_load_status: str,
    errors: Sequence[str],
) -> ProjectCapabilityPipelineResult:
    return ProjectCapabilityPipelineResult(
        status=status,
        source_load_status=source_load_status,
        source_schema_version=None,
        source_content_hash=None,
        source_file_sha256=None,
        project_results=(),
        source_project_count=0,
        source_evidence_fact_count=0,
        source_claim_boundary_count=0,
        source_capability_fact_count=0,
        candidate_count=0,
        matched_evidence_count=0,
        unmatched_evidence_count=0,
        ambiguous_evidence_count=0,
        skipped_evidence_count=0,
        assessment_count=0,
        eligible_assessment_count=0,
        assessment_status_counts={},
        policy_count=0,
        eligible_policy_count=0,
        policy_status_counts={},
        build_result_count=0,
        build_status_counts={},
        capability_fact_count=0,
        memory=None,
        persisted_path=None,
        warnings=(),
        errors=tuple(errors),
        diagnostics={
            "projects_with_capabilities": 0,
            "projects_without_capabilities": 0,
        },
    )


def _aggregate_result(
    *,
    source_memory: ProjectEvidenceMemorySnapshot,
    source_file_sha256: str | None,
    project_executions: Sequence[_ProjectExecution],
    memory: ProjectCapabilityMemory | None,
    errors: Sequence[str] = (),
) -> ProjectCapabilityPipelineResult:
    projects = tuple(item.result for item in project_executions)
    source_evidence_count = sum(len(project.evidence_facts) for project in source_memory.projects)
    source_boundary_count = sum(len(project.claim_boundaries) for project in source_memory.projects)
    source_capability_count = sum(len(project.capability_facts) for project in source_memory.projects)
    warnings = {warning for project in projects for warning in project.warnings}
    if source_capability_count:
        warnings.add("source_capability_facts_ignored")
    eligible_assessment_count = sum(item.eligible_assessment_count for item in projects)
    facts = tuple(fact for execution in project_executions for fact in execution.facts)
    if not eligible_assessment_count:
        warnings.add("no_eligible_capability_assessments")
    if not facts:
        warnings.add("capability_facts_empty")
    if memory is not None:
        warnings.update(memory.diagnostics.warnings)
    all_errors = set(errors) | {error for project in projects for error in project.errors}
    status = "failed" if all_errors else ("ready" if facts else "empty")
    projects_with = sum(bool(item.capability_fact_ids) for item in projects)
    return ProjectCapabilityPipelineResult(
        status=status,
        source_load_status="ready",
        source_schema_version=source_memory.schema_version,
        source_content_hash=source_memory.content_hash,
        source_file_sha256=source_file_sha256,
        project_results=projects,
        source_project_count=len(source_memory.projects),
        source_evidence_fact_count=source_evidence_count,
        source_claim_boundary_count=source_boundary_count,
        source_capability_fact_count=source_capability_count,
        candidate_count=sum(item.candidate_count for item in projects),
        matched_evidence_count=sum(item.matched_evidence_count for item in projects),
        unmatched_evidence_count=sum(item.unmatched_evidence_count for item in projects),
        ambiguous_evidence_count=sum(item.ambiguous_evidence_count for item in projects),
        skipped_evidence_count=sum(item.skipped_evidence_count for item in projects),
        assessment_count=sum(item.assessment_count for item in projects),
        eligible_assessment_count=eligible_assessment_count,
        assessment_status_counts=_merge_count_mappings(tuple(
            item.assessment_status_counts for item in projects
        )),
        policy_count=sum(item.policy_count for item in projects),
        eligible_policy_count=sum(item.eligible_policy_count for item in projects),
        policy_status_counts=_merge_count_mappings(tuple(
            item.policy_status_counts for item in projects
        )),
        build_result_count=sum(item.build_result_count for item in projects),
        build_status_counts=_merge_count_mappings(tuple(
            item.build_status_counts for item in projects
        )),
        capability_fact_count=len(facts) if not all_errors else 0,
        memory=memory if not all_errors else None,
        persisted_path=None,
        warnings=tuple(warnings),
        errors=tuple(all_errors),
        diagnostics={
            "eligible_assessment_count": eligible_assessment_count,
            "eligible_policy_count": sum(item.eligible_policy_count for item in projects),
            "projects_with_capabilities": projects_with,
            "projects_without_capabilities": len(projects) - projects_with,
            "source_capability_fact_count": source_capability_count,
        },
    )


def build_project_capability_pipeline(
    *,
    source_memory: ProjectEvidenceMemorySnapshot,
    source_file_sha256: str | None = None,
) -> ProjectCapabilityPipelineResult:
    """Run the authoritative lifecycle in memory without performing file writes."""

    if not isinstance(source_memory, ProjectEvidenceMemorySnapshot):
        return _source_failure_result(
            status="source_invalid",
            source_load_status="invalid",
            errors=("source_artifact_invalid",),
        )
    validation = evidence_memory_store.validate_project_evidence_memory_snapshot(source_memory)
    if not validation.valid:
        code = (
            "source_hash_mismatch"
            if any(item in validation.errors for item in ("content_hash_mismatch", "invalid_content_hash"))
            else "source_schema_unsupported"
            if "unsupported_schema_version" in validation.errors
            else "source_artifact_invalid"
        )
        return _source_failure_result(
            status="source_invalid",
            source_load_status="invalid",
            errors=(code,),
        )
    if source_file_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", source_file_sha256):
        return _source_failure_result(
            status="source_invalid",
            source_load_status="invalid",
            errors=("source_artifact_invalid",),
        )

    executions = tuple(_process_project(project) for project in sorted(
        source_memory.projects, key=lambda item: (item.project_id.casefold(), item.project_id)
    ))
    if any(item.result.status == "failed" for item in executions):
        return _aggregate_result(
            source_memory=source_memory,
            source_file_sha256=source_file_sha256,
            project_executions=executions,
            memory=None,
        )

    facts = tuple(fact for execution in executions for fact in execution.facts)
    fact_ids = [fact.capability_id for fact in facts]
    project_types = [(fact.project_id, fact.capability_type) for fact in facts]
    if len(fact_ids) != len(set(fact_ids)) or len(project_types) != len(set(project_types)):
        return _aggregate_result(
            source_memory=source_memory,
            source_file_sha256=source_file_sha256,
            project_executions=executions,
            memory=None,
            errors=("duplicate_final_capability_identity",),
        )

    try:
        source_artifact = ProjectCapabilitySourceArtifact(
            schema_version=source_memory.schema_version,
            content_hash=source_memory.content_hash,
            file_sha256=source_file_sha256,
            project_count=len(source_memory.projects),
            evidence_fact_count=sum(len(project.evidence_facts) for project in source_memory.projects),
            claim_boundary_count=sum(len(project.claim_boundaries) for project in source_memory.projects),
        )
        memory = capability_memory_module.build_project_capability_memory(
            source_artifact=source_artifact,
            source_project_ids=tuple(project.project_id for project in source_memory.projects),
            capability_facts=facts,
        )
        if not capability_memory_module.validate_project_capability_memory(memory).valid:
            raise _PipelineInvariantError("memory_validation_failed")
    except Exception:
        return _aggregate_result(
            source_memory=source_memory,
            source_file_sha256=source_file_sha256,
            project_executions=executions,
            memory=None,
            errors=("memory_validation_failed",),
        )
    return _aggregate_result(
        source_memory=source_memory,
        source_file_sha256=source_file_sha256,
        project_executions=executions,
        memory=memory,
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(
            str(right.resolve(strict=False))
        )
    except OSError:
        return os.path.normcase(str(left.absolute())) == os.path.normcase(str(right.absolute()))


def _safe_path_name(path: Path) -> str:
    return path.name if _SAFE_PATH_NAME_RE.fullmatch(path.name) else "project_capability_memory.json"


def run_project_capability_pipeline(
    *,
    source_path: Path | str = evidence_memory_store.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
    persist: bool = False,
    output_path: Path | str | None = None,
) -> ProjectCapabilityPipelineResult:
    """Load, orchestrate, and optionally persist only to an explicit safe path."""

    if persist and output_path is None:
        return _source_failure_result(
            status="failed",
            source_load_status="not_loaded",
            errors=("explicit_output_path_required",),
        )
    if persist and _same_path(Path(output_path), PROJECT_CAPABILITY_MEMORY_PATH):
        return _source_failure_result(
            status="failed",
            source_load_status="not_loaded",
            errors=("real_output_path_forbidden",),
        )

    source = Path(source_path)
    loaded = evidence_memory_store.load_project_evidence_memory(source)
    if loaded.status == "missing":
        return _source_failure_result(
            status="source_missing",
            source_load_status="missing",
            errors=("source_artifact_missing",),
        )
    if loaded.status != "ready" or loaded.snapshot is None:
        code = (
            "source_schema_unsupported" if loaded.status == "unsupported_version"
            else "source_hash_mismatch" if loaded.status == "hash_mismatch"
            else "source_artifact_invalid"
        )
        return _source_failure_result(
            status="source_invalid",
            source_load_status=loaded.status,
            errors=(code,),
        )
    try:
        source_file_sha256 = evidence_memory_module.stable_hash(source.read_bytes())
    except OSError:
        return _source_failure_result(
            status="source_invalid",
            source_load_status="invalid",
            errors=("source_artifact_invalid",),
        )
    result = build_project_capability_pipeline(
        source_memory=loaded.snapshot,
        source_file_sha256=source_file_sha256,
    )
    if not persist or result.status not in {"ready", "empty"} or result.memory is None:
        return result

    destination = Path(output_path)
    report = capability_memory_module.persist_project_capability_memory(result.memory, destination)
    if report.status not in {"created", "updated", "unchanged"} or not report.round_trip_validated:
        return replace(
            result,
            status="failed",
            memory=None,
            errors=_codes((*result.errors, "persistence_validation_failed")),
        )
    reloaded = capability_memory_module.load_project_capability_memory(destination)
    expected_status = "ready" if result.memory.capability_facts else "empty"
    if reloaded.status != expected_status or reloaded.memory != result.memory:
        return replace(
            result,
            status="failed",
            memory=None,
            errors=_codes((*result.errors, "persistence_validation_failed")),
        )
    return replace(result, persisted_path=_safe_path_name(destination))


__all__ = [
    "PROJECT_CAPABILITY_PIPELINE_STATUSES",
    "PROJECT_CAPABILITY_PROJECT_PIPELINE_STATUSES",
    "ProjectCapabilityPipelineResult",
    "ProjectCapabilityProjectPipelineResult",
    "build_project_capability_pipeline",
    "run_project_capability_pipeline",
]
