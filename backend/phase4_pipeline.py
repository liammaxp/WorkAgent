"""Authoritative orchestration for the accepted Phase 4 memory stages.

The feature is default-off through ``USE_PHASE4_PROJECT_MEMORY``.  This module
coordinates the public Step 2--9 interfaces; it contains no alternative
evidence, capability, claim, or persistence rules and performs no work when it
is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Any, Mapping

from backend import phase4_capability_extractor as capability_extractor
from backend import phase4_capability_taxonomy as capability_taxonomy
from backend import phase4_claim_boundary as claim_boundary
from backend import phase4_evidence_normalizer as evidence_normalizer
from backend import phase4_evidence_scoring as evidence_scoring
from backend import phase4_evidence_synthesizer as evidence_synthesizer
from backend import phase4_input_adapter as input_adapter
from backend import phase4_project_memory as project_memory
from backend.phase4_models import (
    ClaimSubjectType,
    Phase4PipelineWarning,
    WarningSeverity,
)


PHASE4_PROJECT_MEMORY_FLAG = "USE_PHASE4_PROJECT_MEMORY"
ENABLED_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
DISABLED_FLAG_VALUES = frozenset({"", "0", "false", "no", "off"})
MAX_PIPELINE_WARNINGS = 100
MAX_PIPELINE_ERRORS = 100
DEFAULT_INSPECT_SAMPLE_LIMIT = 5
MAX_INSPECT_SAMPLE_LIMIT = 20
MAX_INSPECT_PROJECT_ID_LENGTH = 300

PHASE4_PIPELINE_STAGE_ORDER = (
    "load_inputs",
    "normalize_dedupe",
    "synthesize_evidence",
    "score_evidence",
    "validate_taxonomy",
    "extract_signals",
    "extract_capabilities",
    "build_claim_boundaries",
    "build_project_memory",
    "persist_project_memory",
    "round_trip_validate",
)
PIPELINE_STAGE_STATUSES = frozenset({
    "not_started", "skipped", "ready", "empty", "unchanged", "degraded", "error",
})
PIPELINE_STATUSES = frozenset({"disabled", "empty", "ready", "degraded", "unchanged", "error"})
NEXT_RECOMMENDED_ACTIONS = frozenset({
    "enable_phase4",
    "provide_safe_source_data",
    "review_optional_source_mapping",
    "use_existing_artifact",
    "pipeline_ready",
    "retry_pipeline",
    "repair_invalid_artifact",
})

_WARNING_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_BUILD_LOCK = threading.Lock()
_WARNING_MESSAGES = {
    "artifact_unchanged": "The validated Phase 4 artifact already matches the current snapshot.",
    "capability_facts_empty": "No emitted capability facts were available; this is a valid bounded result.",
    "claim_budget_truncated": "Claim budgets deterministically truncated some candidate claims.",
    "compact_fact_project_mapping_missing": "A compact-fact record had no established project mapping.",
    "contextual_only_project": "At least one project contains only bounded contextual support.",
    "free_text_source_deferred": "Unstructured free-text ingestion remains outside this pipeline contract.",
    "invalid_feature_flag": "The Phase 4 feature flag value is unsupported and was treated as disabled.",
    "low_quality_facts_excluded": "Low-quality evidence was excluded from affirmative claims.",
    "metric_evidence_empty": "No supported resume metric claims were emitted.",
    "missing_project_id": "One or more structured source records had no established project mapping.",
    "optional_phase3_artifact_missing": "The optional Phase 3 artifact was not available.",
    "optional_source_missing": "An optional structured source artifact was not available.",
    "persistence_skipped": "A valid snapshot was built without persisting it.",
    "record_validation_failed": "One or more structured source records failed bounded validation.",
    "source_file_missing": "An optional structured source artifact was not available.",
    "source_json_invalid": "An optional structured source artifact contained invalid JSON.",
    "source_schema_invalid": "An optional structured source artifact had an invalid schema.",
    "taxonomy_warning": "The capability taxonomy emitted a bounded non-fatal warning.",
    "unsupported_source_schema": "An optional structured source artifact used an unsupported schema.",
    "unsafe_impact_dropped": "Unsupported impact content was excluded during evidence synthesis.",
}


@dataclass(frozen=True)
class Phase4PipelineStageResult:
    stage: str
    status: str
    input_count: int = 0
    output_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    details: Mapping[str, int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in PHASE4_PIPELINE_STAGE_ORDER:
            raise ValueError("unknown Phase 4 pipeline stage")
        if self.status not in PIPELINE_STAGE_STATUSES:
            raise ValueError("unknown Phase 4 stage status")
        for name in ("input_count", "output_count", "warning_count", "error_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.details, Mapping):
            raise TypeError("stage details must be a mapping")
        safe: dict[str, int | str | bool] = {}
        for key, value in self.details.items():
            if not isinstance(key, str) or not _WARNING_CODE_RE.fullmatch(key):
                raise ValueError("stage detail keys must be safe identifiers")
            if not isinstance(value, (bool, int, str)):
                raise TypeError("stage detail values must be integers, strings, or booleans")
            if isinstance(value, str) and len(value) > 100:
                raise ValueError("stage detail strings must be bounded")
            safe[key] = value
        object.__setattr__(self, "details", MappingProxyType(dict(sorted(safe.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Phase4ProjectMemoryPipelineResult:
    status: str
    enabled: bool
    persisted: bool
    persistence_status: str | None
    artifact_path: str | None
    schema_version: str | None
    content_hash: str | None
    input_count: int
    normalized_input_count: int
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int
    project_count: int
    allowed_claim_count: int
    forbidden_claim_count: int
    claim_truncation_count: int
    previous_artifact_preserved: bool
    stages: tuple[Phase4PipelineStageResult, ...]
    warnings: tuple[Phase4PipelineWarning, ...]
    errors: tuple[str, ...]
    next_recommended_action: str

    def __post_init__(self) -> None:
        if self.status not in PIPELINE_STATUSES:
            raise ValueError("unknown Phase 4 pipeline status")
        if self.next_recommended_action not in NEXT_RECOMMENDED_ACTIONS:
            raise ValueError("unknown next recommended action")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "enabled": self.enabled,
            "persisted": self.persisted,
            "persistence_status": self.persistence_status,
            "artifact_path": self.artifact_path,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "input_count": self.input_count,
            "normalized_input_count": self.normalized_input_count,
            "evidence_fact_count": self.evidence_fact_count,
            "capability_fact_count": self.capability_fact_count,
            "claim_boundary_count": self.claim_boundary_count,
            "project_count": self.project_count,
            "allowed_claim_count": self.allowed_claim_count,
            "forbidden_claim_count": self.forbidden_claim_count,
            "claim_truncation_count": self.claim_truncation_count,
            "previous_artifact_preserved": self.previous_artifact_preserved,
            "stages": [stage.to_dict() for stage in self.stages],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": list(self.errors),
            "next_recommended_action": self.next_recommended_action,
        }


# Backward-friendly public name for callers expecting a pipeline result.
Phase4PipelineResult = Phase4ProjectMemoryPipelineResult


@dataclass(frozen=True)
class Phase4PipelineHealth:
    enabled: bool
    status: str
    artifact_status: str
    schema_version: str | None = None
    content_hash: str | None = None
    hash_valid: bool = False
    project_count: int = 0
    evidence_fact_count: int = 0
    capability_fact_count: int = 0
    claim_boundary_count: int = 0
    warning_codes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    next_recommended_action: str = "enable_phase4"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "artifact_status": self.artifact_status,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "hash_valid": self.hash_valid,
            "project_count": self.project_count,
            "evidence_fact_count": self.evidence_fact_count,
            "capability_fact_count": self.capability_fact_count,
            "claim_boundary_count": self.claim_boundary_count,
            "warning_codes": list(self.warning_codes),
            "errors": list(self.errors),
            "next_recommended_action": self.next_recommended_action,
        }


@dataclass(frozen=True)
class Phase4ProjectInspectSummary:
    project_id: str
    project_name: str
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int
    evidence_boundary_count: int
    capability_boundary_count: int
    project_boundary_count: int
    accepted_fact_count: int
    supporting_fact_count: int
    weak_fact_count: int
    rejected_fact_count: int
    warning_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "evidence_fact_count": self.evidence_fact_count,
            "capability_fact_count": self.capability_fact_count,
            "claim_boundary_count": self.claim_boundary_count,
            "evidence_boundary_count": self.evidence_boundary_count,
            "capability_boundary_count": self.capability_boundary_count,
            "project_boundary_count": self.project_boundary_count,
            "accepted_fact_count": self.accepted_fact_count,
            "supporting_fact_count": self.supporting_fact_count,
            "weak_fact_count": self.weak_fact_count,
            "rejected_fact_count": self.rejected_fact_count,
            "warning_codes": list(self.warning_codes),
        }


@dataclass(frozen=True)
class Phase4ProjectInspect:
    enabled: bool
    status: str
    schema_version: str | None
    content_hash: str | None
    project_id: str | None
    sample_limit: int
    project_count: int
    returned_project_count: int
    projects: tuple[Phase4ProjectInspectSummary, ...] = ()
    warning_codes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "project_id": self.project_id,
            "sample_limit": self.sample_limit,
            "project_count": self.project_count,
            "returned_project_count": self.returned_project_count,
            "projects": [project.to_dict() for project in self.projects],
            "warning_codes": list(self.warning_codes),
            "errors": list(self.errors),
        }


class _PipelineStageError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _flag_state(environ: Mapping[str, str] | None = None) -> tuple[bool, bool]:
    source = os.environ if environ is None else environ
    raw = source.get(PHASE4_PROJECT_MEMORY_FLAG, "")
    normalized = raw.strip().casefold() if isinstance(raw, str) else ""
    if normalized in ENABLED_FLAG_VALUES:
        return True, True
    if normalized in DISABLED_FLAG_VALUES:
        return False, True
    return False, False


def is_phase4_project_memory_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _flag_state(environ)[0]


def _safe_artifact_path(path: Path) -> str:
    if not path.is_absolute():
        parts = [part for part in path.parts if part not in ("", ".")]
        return path.name if ".." in parts else Path(*parts).as_posix()
    try:
        return path.relative_to(project_memory.ROOT_DIR).as_posix()
    except ValueError:
        return path.name


def _safe_warning(code: str) -> Phase4PipelineWarning:
    safe_code = code if _WARNING_CODE_RE.fullmatch(code) else "pipeline_warning"
    message = _WARNING_MESSAGES.get(safe_code, "A Phase 4 stage emitted a bounded warning.")
    return Phase4PipelineWarning(
        code=safe_code,
        message=message,
        severity=WarningSeverity.WARNING,
    )


def _source_type_details(inputs: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in inputs:
        normalized = re.sub(r"[^a-z0-9]+", "_", item.input_type.casefold()).strip("_")
        if normalized:
            key = f"source_type_{normalized}_count"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items())[:50])


def _warning_tuple(codes: set[str]) -> tuple[Phase4PipelineWarning, ...]:
    return tuple(_safe_warning(code) for code in sorted(codes)[:MAX_PIPELINE_WARNINGS])


def _next_action(status: str, *, previous_valid: bool, invalid_existing: bool = False) -> str:
    if status == "disabled":
        return "enable_phase4"
    if status == "empty":
        return "provide_safe_source_data"
    if status == "unchanged":
        return "use_existing_artifact"
    if status in {"ready", "degraded"}:
        return "pipeline_ready"
    if previous_valid:
        return "use_existing_artifact"
    if invalid_existing:
        return "repair_invalid_artifact"
    return "retry_pipeline"


def _error_code(stage: str, error: Exception) -> str:
    if isinstance(error, _PipelineStageError):
        return error.code
    if isinstance(error, evidence_normalizer.Phase4IntegrityError):
        return "normalization_integrity_conflict"
    if isinstance(error, project_memory.Phase4ProjectMemoryIntegrityError):
        return "project_memory_integrity_error"
    if isinstance(error, (TypeError, ValueError)):
        return f"{stage}_validation_failed"
    return "internal_pipeline_error"


def _skipped_stages(existing: list[Phase4PipelineStageResult]) -> None:
    completed = {stage.stage for stage in existing}
    for name in PHASE4_PIPELINE_STAGE_ORDER:
        if name not in completed:
            existing.append(Phase4PipelineStageResult(stage=name, status="skipped"))


def _result(
    *,
    status: str,
    enabled: bool,
    destination: Path,
    stages: list[Phase4PipelineStageResult],
    warning_codes: set[str],
    errors: set[str],
    counts: Mapping[str, int],
    previous_artifact_preserved: bool,
    persisted: bool = False,
    persistence_status: str | None = None,
    schema_version: str | None = None,
    content_hash: str | None = None,
    invalid_existing: bool = False,
) -> Phase4ProjectMemoryPipelineResult:
    _skipped_stages(stages)
    return Phase4ProjectMemoryPipelineResult(
        status=status,
        enabled=enabled,
        persisted=persisted,
        persistence_status=persistence_status,
        artifact_path=_safe_artifact_path(destination),
        schema_version=schema_version,
        content_hash=content_hash,
        input_count=counts.get("input_count", 0),
        normalized_input_count=counts.get("normalized_input_count", 0),
        evidence_fact_count=counts.get("evidence_fact_count", 0),
        capability_fact_count=counts.get("capability_fact_count", 0),
        claim_boundary_count=counts.get("claim_boundary_count", 0),
        project_count=counts.get("project_count", 0),
        allowed_claim_count=counts.get("allowed_claim_count", 0),
        forbidden_claim_count=counts.get("forbidden_claim_count", 0),
        claim_truncation_count=counts.get("claim_truncation_count", 0),
        previous_artifact_preserved=previous_artifact_preserved,
        stages=tuple(stages),
        warnings=_warning_tuple(warning_codes),
        errors=tuple(sorted(errors)[:MAX_PIPELINE_ERRORS]),
        next_recommended_action=_next_action(
            status,
            previous_valid=previous_artifact_preserved,
            invalid_existing=invalid_existing,
        ),
    )


def run_phase4_project_memory_pipeline(
    *,
    source_paths: input_adapter.Phase4InputSourcePaths | None = None,
    output_path: str | Path | None = None,
    persist: bool = True,
    environ: Mapping[str, str] | None = None,
) -> Phase4ProjectMemoryPipelineResult:
    """Run accepted Steps 2--9 in order when the default-off flag is enabled."""

    destination = Path(output_path) if output_path is not None else project_memory.DEFAULT_PHASE4_PROJECT_MEMORY_PATH
    enabled, recognized_flag = _flag_state(environ)
    warning_codes: set[str] = set()
    errors: set[str] = set()
    counts: dict[str, int] = {}
    stages: list[Phase4PipelineStageResult] = []
    if not recognized_flag:
        warning_codes.add("invalid_feature_flag")
    if not enabled:
        return _result(
            status="disabled",
            enabled=False,
            destination=destination,
            stages=stages,
            warning_codes=warning_codes,
            errors=errors,
            counts=counts,
            previous_artifact_preserved=False,
        )

    with _BUILD_LOCK:
        try:
            existing = project_memory.load_phase4_project_memory(destination)
        except Exception as error:
            code = _error_code("load_inputs", error)
            errors.add(code)
            stages.append(Phase4PipelineStageResult(
                stage="load_inputs", status="error", error_count=1,
                details={"error_code": code},
            ))
            return _result(
                status="error", enabled=True, destination=destination, stages=stages,
                warning_codes=warning_codes, errors=errors, counts=counts,
                previous_artifact_preserved=False,
            )
        previous_valid = existing.status == "ready"
        invalid_existing = existing.status not in {"missing", "ready"}
        current_stage = "load_inputs"
        current_input_count = 0
        snapshot: project_memory.Phase4ProjectMemorySnapshot | None = None
        persistence_status: str | None = None
        try:
            paths = source_paths or input_adapter.Phase4InputSourcePaths()
            inputs, adapter_warnings = input_adapter.load_phase4_inputs(source_paths=paths)
            counts["input_count"] = len(inputs)
            for warning in adapter_warnings:
                warning_codes.add(warning.code)
                if warning.code == "missing_project_id":
                    warning_codes.add("compact_fact_project_mapping_missing")
            if paths.phase3_memory_path is not None and not Path(paths.phase3_memory_path).exists():
                warning_codes.add("optional_phase3_artifact_missing")
            if any(warning.code == "source_file_missing" for warning in adapter_warnings):
                warning_codes.add("optional_source_missing")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="empty" if not inputs else ("degraded" if adapter_warnings else "ready"),
                input_count=0,
                output_count=len(inputs),
                warning_count=len(adapter_warnings),
                details={
                    "source_type_count": len({item.input_type for item in inputs}),
                    "skipped_count": len(adapter_warnings),
                    "validation_failure_count": sum(
                        warning.code == "record_validation_failed" for warning in adapter_warnings
                    ),
                    **_source_type_details(inputs),
                },
            ))
            if not inputs:
                return _result(
                    status="empty", enabled=True, destination=destination, stages=stages,
                    warning_codes=warning_codes, errors=errors, counts=counts,
                    previous_artifact_preserved=previous_valid,
                    invalid_existing=invalid_existing,
                )

            current_stage = "normalize_dedupe"
            current_input_count = len(inputs)
            normalized, dedupe_report = evidence_normalizer.dedupe_phase4_inputs(inputs)
            counts["normalized_input_count"] = len(normalized)
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="ready",
                input_count=len(inputs),
                output_count=len(normalized),
                details={
                    "normalized_count": dedupe_report.normalized_count,
                    "exact_duplicates_removed": dedupe_report.exact_duplicates_removed,
                    "repeated_source_records_removed": dedupe_report.repeated_source_records_removed,
                    "duplicate_source_refs_removed": dedupe_report.duplicate_source_refs_removed,
                    "duplicate_signal_values_removed": dedupe_report.duplicate_signal_values_removed,
                    "integrity_conflict_count": 0,
                },
            ))

            current_stage = "synthesize_evidence"
            current_input_count = len(normalized)
            facts, synthesis_report = evidence_synthesizer.synthesize_phase4_evidence_facts(normalized)
            counts["evidence_fact_count"] = len(facts)
            if synthesis_report.unsafe_impact_dropped_count:
                warning_codes.add("unsafe_impact_dropped")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="empty" if not facts else "ready",
                input_count=len(normalized),
                output_count=len(facts),
                warning_count=int(bool(synthesis_report.unsafe_impact_dropped_count)),
                details={
                    "accepted_count": synthesis_report.accepted_count,
                    "supporting_count": synthesis_report.supporting_count,
                    "weak_count": synthesis_report.weak_count,
                    "rejected_count": synthesis_report.rejected_count,
                    "missing_mechanism_count": synthesis_report.missing_mechanism_count,
                    "missing_implementation_count": synthesis_report.missing_implementation_count,
                    "unsafe_impact_dropped_count": synthesis_report.unsafe_impact_dropped_count,
                },
            ))
            if not facts:
                return _result(
                    status="empty", enabled=True, destination=destination, stages=stages,
                    warning_codes=warning_codes, errors=errors, counts=counts,
                    previous_artifact_preserved=previous_valid,
                    invalid_existing=invalid_existing,
                )

            current_stage = "score_evidence"
            current_input_count = len(facts)
            scored, scoring_report = evidence_scoring.score_phase4_evidence_facts(facts)
            counts["evidence_fact_count"] = len(scored)
            if scoring_report.weak_value_count or scoring_report.rejected_value_count:
                warning_codes.add("low_quality_facts_excluded")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status=(
                    "degraded"
                    if scoring_report.weak_value_count or scoring_report.rejected_value_count
                    else "ready"
                ),
                input_count=len(facts),
                output_count=len(scored),
                warning_count=int(bool(scoring_report.weak_value_count or scoring_report.rejected_value_count)),
                details={
                    "high_value_count": scoring_report.high_value_count,
                    "supporting_value_count": scoring_report.supporting_value_count,
                    "weak_value_count": scoring_report.weak_value_count,
                    "rejected_value_count": scoring_report.rejected_value_count,
                    "minimum_score": scoring_report.minimum_score,
                    "maximum_score": scoring_report.maximum_score,
                    "median_score_x100": int(round(scoring_report.median_score * 100)),
                    "penalty_count": (
                        scoring_report.unsupported_claim_penalty_count
                        + scoring_report.generic_content_penalty_count
                    ),
                    "unsupported_claim_penalty_count": scoring_report.unsupported_claim_penalty_count,
                    "generic_content_penalty_count": scoring_report.generic_content_penalty_count,
                },
            ))

            current_stage = "validate_taxonomy"
            current_input_count = len(scored)
            taxonomy_report = capability_taxonomy.validate_phase4_capability_taxonomy()
            if not taxonomy_report.valid:
                raise _PipelineStageError("taxonomy_validation_failed")
            if taxonomy_report.warnings:
                warning_codes.add("taxonomy_warning")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="degraded" if taxonomy_report.warnings else "ready",
                input_count=taxonomy_report.capability_count,
                output_count=taxonomy_report.capability_count,
                warning_count=len(taxonomy_report.warnings),
                details={
                    "capability_definition_count": taxonomy_report.capability_count,
                    "signal_count": taxonomy_report.signal_count,
                    "alias_count": taxonomy_report.alias_count,
                },
            ))

            current_stage = "extract_signals"
            current_input_count = len(scored)
            _signal_results, signal_report = capability_extractor.extract_phase4_fact_signals_many(scored)
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="ready",
                input_count=len(scored),
                output_count=signal_report.signal_binding_count,
                details={
                    "facts_with_signals": signal_report.facts_with_signals,
                    "facts_without_signals": signal_report.facts_without_signals,
                    "signal_binding_count": signal_report.signal_binding_count,
                    "unique_signal_count": signal_report.unique_signal_count,
                    "ambiguous_rejection_count": signal_report.ambiguous_match_rejected_count,
                },
            ))

            current_stage = "extract_capabilities"
            current_input_count = len(scored)
            grouped_capabilities, capability_report = capability_extractor.extract_phase4_capabilities_by_project(scored)
            capabilities = [
                capability
                for project_id in sorted(grouped_capabilities)
                for capability in grouped_capabilities[project_id]
            ]
            counts["capability_fact_count"] = len(capabilities)
            if not capabilities:
                warning_codes.add("capability_facts_empty")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="degraded" if not capabilities else "ready",
                input_count=len(scored),
                output_count=len(capabilities),
                warning_count=int(not capabilities),
                details={
                    "candidate_count": capability_report.capability_candidates_evaluated,
                    "emitted_count": capability_report.capabilities_emitted,
                    "missing_required_group_count": capability_report.missing_required_group_count,
                    "insufficient_direct_fact_count": capability_report.insufficient_direct_fact_count,
                    "high_risk_blocked_count": capability_report.high_risk_blocked_count,
                },
            ))

            current_stage = "build_claim_boundaries"
            current_input_count = len(scored) + len(capabilities)
            project_boundaries, boundary_report = claim_boundary.build_phase4_claim_boundaries_by_project(
                scored, capabilities
            )
            evidence_boundaries = [
                boundary
                for fact in scored
                if (boundary := claim_boundary.build_phase4_evidence_claim_boundary(fact)) is not None
            ]
            evidence_by_id = {fact.evidence_fact_id: fact for fact in scored}
            capability_by_id = {capability.capability_id: capability for capability in capabilities}
            capability_boundaries = [
                boundary
                for capability in capabilities
                if (boundary := claim_boundary.build_phase4_capability_claim_boundary(
                    capability, evidence_facts_by_id=evidence_by_id
                )) is not None
            ]
            all_boundaries = [
                *evidence_boundaries,
                *capability_boundaries,
                *project_boundaries.values(),
            ]
            validation_failures = sum(
                not claim_boundary.validate_phase4_claim_boundary(
                    boundary,
                    evidence_facts_by_id=evidence_by_id,
                    capability_facts_by_id=capability_by_id,
                ).valid
                for boundary in all_boundaries
            )
            if (
                validation_failures
                or boundary_report.conflict_count
                or boundary_report.project_mismatch_count
                or len(evidence_boundaries) != boundary_report.evidence_boundaries_created
                or len(capability_boundaries) != boundary_report.capability_boundaries_created
            ):
                raise _PipelineStageError("boundary_validation_failed")
            counts.update({
                "claim_boundary_count": len(all_boundaries),
                "allowed_claim_count": boundary_report.allowed_claim_count,
                "forbidden_claim_count": boundary_report.forbidden_claim_count,
                "claim_truncation_count": boundary_report.truncated_claim_count,
            })
            if boundary_report.truncated_claim_count:
                warning_codes.add("claim_budget_truncated")
            if boundary_report.metric_claim_count == 0:
                warning_codes.add("metric_evidence_empty")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="degraded" if boundary_report.truncated_claim_count else "ready",
                input_count=current_input_count,
                output_count=len(all_boundaries),
                warning_count=(
                    int(bool(boundary_report.truncated_claim_count))
                    + int(boundary_report.metric_claim_count == 0)
                ),
                details={
                    "evidence_boundary_count": len(evidence_boundaries),
                    "capability_boundary_count": len(capability_boundaries),
                    "project_boundary_count": len(project_boundaries),
                    "allowed_claim_count": boundary_report.allowed_claim_count,
                    "forbidden_claim_count": boundary_report.forbidden_claim_count,
                    "truncation_count": boundary_report.truncated_claim_count,
                    "validation_failure_count": validation_failures,
                },
            ))

            project_ids = sorted({fact.project_id for fact in scored})
            projects_with_truncation = sum(
                claim_boundary.build_phase4_claim_boundaries_by_project(
                    [fact for fact in scored if fact.project_id == project_id],
                    [capability for capability in capabilities if capability.project_id == project_id],
                )[1].truncated_claim_count > 0
                for project_id in project_ids
            )
            diagnostics = project_memory.Phase4ProjectMemoryDiagnostics.from_claim_boundary_report(
                boundary_report,
                projects_with_truncation=projects_with_truncation,
            )

            current_stage = "build_project_memory"
            current_input_count = len(scored) + len(capabilities) + len(all_boundaries)
            memories, memory_report = project_memory.build_phase4_project_memories(
                scored, capabilities, all_boundaries, diagnostics=diagnostics
            )
            warning_codes.update(memory_report.warnings)
            snapshot = project_memory.build_phase4_project_memory_snapshot(
                memories, diagnostics=diagnostics
            )
            snapshot_validation = project_memory.validate_phase4_project_memory_snapshot(snapshot)
            if not snapshot_validation.valid:
                raise _PipelineStageError("snapshot_validation_failed")
            serialized = project_memory.serialize_phase4_project_memory_snapshot(snapshot)
            counts["project_count"] = len(memories)
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="ready",
                input_count=current_input_count,
                output_count=len(memories),
                warning_count=len(memory_report.warnings),
                details={
                    "project_count": len(memories),
                    "evidence_fact_count": memory_report.evidence_fact_count,
                    "capability_fact_count": memory_report.capability_fact_count,
                    "claim_boundary_count": memory_report.claim_boundary_count,
                    "serialized_size": len(serialized),
                },
            ))

            current_stage = "persist_project_memory"
            current_input_count = len(memories)
            if not persist:
                warning_codes.add("persistence_skipped")
                stages.append(Phase4PipelineStageResult(
                    stage=current_stage,
                    status="skipped",
                    input_count=len(memories),
                    output_count=0,
                    warning_count=1,
                    details={"persist_requested": False},
                ))
                stages.append(Phase4PipelineStageResult(
                    stage="round_trip_validate",
                    status="skipped",
                    input_count=0,
                    output_count=0,
                ))
                final_status = "degraded" if warning_codes else "ready"
                return _result(
                    status=final_status, enabled=True, destination=destination, stages=stages,
                    warning_codes=warning_codes, errors=errors, counts=counts,
                    previous_artifact_preserved=previous_valid,
                    schema_version=snapshot.schema_version, content_hash=snapshot.content_hash,
                    invalid_existing=invalid_existing,
                )

            persistence_report = project_memory.persist_phase4_project_memory(snapshot, destination)
            persistence_status = persistence_report.status
            if persistence_status == "failed":
                raise _PipelineStageError("persistence_failed")
            if persistence_status == "unchanged":
                warning_codes.add("artifact_unchanged")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="unchanged" if persistence_status == "unchanged" else "ready",
                input_count=len(memories),
                output_count=len(memories),
                warning_count=int(persistence_status == "unchanged"),
                details={
                    "persistence_status": persistence_status,
                    "bytes_written": persistence_report.bytes_written,
                    "previous_artifact_preserved": persistence_report.previous_artifact_preserved,
                },
            ))

            current_stage = "round_trip_validate"
            current_input_count = len(memories)
            if not persistence_report.round_trip_validated:
                raise _PipelineStageError("round_trip_validation_failed")
            stages.append(Phase4PipelineStageResult(
                stage=current_stage,
                status="unchanged" if persistence_status == "unchanged" else "ready",
                input_count=len(memories),
                output_count=len(memories),
                details={"round_trip_validated": True},
            ))
            final_status = (
                "unchanged"
                if persistence_status == "unchanged"
                else ("degraded" if warning_codes else "ready")
            )
            return _result(
                status=final_status,
                enabled=True,
                destination=destination,
                stages=stages,
                warning_codes=warning_codes,
                errors=errors,
                counts=counts,
                previous_artifact_preserved=persistence_report.previous_artifact_preserved,
                persisted=True,
                persistence_status=persistence_status,
                schema_version=snapshot.schema_version,
                content_hash=snapshot.content_hash,
                invalid_existing=invalid_existing,
            )
        except Exception as error:
            code = _error_code(current_stage, error)
            errors.add(code)
            if not any(stage.stage == current_stage for stage in stages):
                stages.append(Phase4PipelineStageResult(
                    stage=current_stage,
                    status="error",
                    input_count=current_input_count,
                    output_count=0,
                    error_count=1,
                    details={"error_code": code},
                ))
            return _result(
                status="error",
                enabled=True,
                destination=destination,
                stages=stages,
                warning_codes=warning_codes,
                errors=errors,
                counts=counts,
                previous_artifact_preserved=previous_valid,
                persisted=False,
                persistence_status=persistence_status,
                schema_version=snapshot.schema_version if snapshot is not None else None,
                content_hash=snapshot.content_hash if snapshot is not None else None,
                invalid_existing=invalid_existing,
            )


def get_phase4_pipeline_health(
    *,
    output_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Phase4PipelineHealth:
    """Read and validate the existing artifact without running or writing the pipeline."""

    enabled, recognized = _flag_state(environ)
    flag_warnings = () if recognized else ("invalid_feature_flag",)
    if not enabled:
        return Phase4PipelineHealth(
            enabled=False,
            status="disabled",
            artifact_status="not_checked",
            warning_codes=flag_warnings,
            next_recommended_action="enable_phase4",
        )
    destination = Path(output_path) if output_path is not None else project_memory.DEFAULT_PHASE4_PROJECT_MEMORY_PATH
    try:
        loaded = project_memory.load_phase4_project_memory(destination)
    except Exception:
        return Phase4PipelineHealth(
            enabled=True,
            status="error",
            artifact_status="error",
            errors=("artifact_load_failed",),
            next_recommended_action="retry_pipeline",
        )
    if loaded.status == "missing":
        return Phase4PipelineHealth(
            enabled=True,
            status="missing",
            artifact_status="missing",
            warning_codes=("artifact_missing",),
            next_recommended_action="retry_pipeline",
        )
    if loaded.status != "ready" or loaded.snapshot is None:
        return Phase4PipelineHealth(
            enabled=True,
            status="invalid",
            artifact_status=loaded.status,
            warning_codes=tuple(sorted(set(loaded.warnings))[:MAX_PIPELINE_WARNINGS]),
            errors=("invalid_artifact",),
            next_recommended_action="repair_invalid_artifact",
        )
    snapshot = loaded.snapshot
    warnings = tuple(sorted(set(snapshot.diagnostics.warnings))[:MAX_PIPELINE_WARNINGS])
    status = "degraded" if warnings else "ready"
    return Phase4PipelineHealth(
        enabled=True,
        status=status,
        artifact_status="ready",
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        hash_valid=loaded.validation.valid,
        project_count=len(snapshot.projects),
        evidence_fact_count=sum(len(project.evidence_facts) for project in snapshot.projects),
        capability_fact_count=sum(len(project.capability_facts) for project in snapshot.projects),
        claim_boundary_count=sum(len(project.claim_boundaries) for project in snapshot.projects),
        warning_codes=warnings,
        next_recommended_action="pipeline_ready",
    )


def _safe_sample_limit(value: int | str | None) -> int:
    try:
        parsed = int(value if value is not None else DEFAULT_INSPECT_SAMPLE_LIMIT)
    except (TypeError, ValueError):
        parsed = DEFAULT_INSPECT_SAMPLE_LIMIT
    return max(0, min(parsed, MAX_INSPECT_SAMPLE_LIMIT))


def _inspect_summary(memory: Any) -> Phase4ProjectInspectSummary:
    boundaries = memory.claim_boundaries
    return Phase4ProjectInspectSummary(
        project_id=memory.project_id,
        project_name=memory.project_name,
        evidence_fact_count=len(memory.evidence_facts),
        capability_fact_count=len(memory.capability_facts),
        claim_boundary_count=len(boundaries),
        evidence_boundary_count=sum(
            boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT for boundary in boundaries
        ),
        capability_boundary_count=sum(
            boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT for boundary in boundaries
        ),
        project_boundary_count=sum(
            boundary.subject_type is ClaimSubjectType.PROJECT for boundary in boundaries
        ),
        accepted_fact_count=memory.quality_summary["accepted_count"],
        supporting_fact_count=memory.quality_summary["supporting_count"],
        weak_fact_count=memory.quality_summary["weak_count"],
        rejected_fact_count=memory.quality_summary["rejected_count"],
        warning_codes=tuple(sorted({warning.code for warning in memory.warnings})),
    )


def get_phase4_project_inspect(
    *,
    output_path: str | Path | None = None,
    project_id: str | None = None,
    sample_limit: int | str = DEFAULT_INSPECT_SAMPLE_LIMIT,
    environ: Mapping[str, str] | None = None,
) -> Phase4ProjectInspect:
    """Return bounded project/count metadata; never return evidence or claims."""

    limit = _safe_sample_limit(sample_limit)
    requested_project_id = " ".join(project_id.split()) if isinstance(project_id, str) else None
    if requested_project_id is not None:
        requested_project_id = requested_project_id[:MAX_INSPECT_PROJECT_ID_LENGTH]
    requested_project_id = requested_project_id or None
    enabled, recognized = _flag_state(environ)
    flag_warnings = () if recognized else ("invalid_feature_flag",)
    if not enabled:
        return Phase4ProjectInspect(
            enabled=False,
            status="disabled",
            schema_version=None,
            content_hash=None,
            project_id=requested_project_id,
            sample_limit=limit,
            project_count=0,
            returned_project_count=0,
            warning_codes=flag_warnings,
        )
    destination = Path(output_path) if output_path is not None else project_memory.DEFAULT_PHASE4_PROJECT_MEMORY_PATH
    try:
        loaded = project_memory.load_phase4_project_memory(destination)
    except Exception:
        return Phase4ProjectInspect(
            enabled=True, status="error", schema_version=None, content_hash=None,
            project_id=requested_project_id, sample_limit=limit, project_count=0,
            returned_project_count=0, errors=("artifact_load_failed",),
        )
    if loaded.status == "missing":
        return Phase4ProjectInspect(
            enabled=True, status="missing", schema_version=None, content_hash=None,
            project_id=requested_project_id, sample_limit=limit, project_count=0,
            returned_project_count=0, warning_codes=("artifact_missing",),
        )
    if loaded.status != "ready" or loaded.snapshot is None:
        return Phase4ProjectInspect(
            enabled=True, status="invalid", schema_version=None, content_hash=None,
            project_id=requested_project_id, sample_limit=limit, project_count=0,
            returned_project_count=0,
            warning_codes=tuple(sorted(set(loaded.warnings))[:MAX_PIPELINE_WARNINGS]),
            errors=("invalid_artifact",),
        )
    snapshot = loaded.snapshot
    selected = [
        memory for memory in snapshot.projects
        if requested_project_id is None or memory.project_id == requested_project_id
    ]
    selected.sort(key=lambda memory: (memory.project_id, memory.project_memory_id))
    summaries = tuple(_inspect_summary(memory) for memory in selected[:limit])
    warnings = tuple(sorted(set(snapshot.diagnostics.warnings))[:MAX_PIPELINE_WARNINGS])
    return Phase4ProjectInspect(
        enabled=True,
        status="degraded" if warnings else "ready",
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        project_id=requested_project_id,
        sample_limit=limit,
        project_count=len(selected),
        returned_project_count=len(summaries),
        projects=summaries,
        warning_codes=warnings,
    )


__all__ = [
    "DEFAULT_INSPECT_SAMPLE_LIMIT",
    "ENABLED_FLAG_VALUES",
    "MAX_INSPECT_SAMPLE_LIMIT",
    "MAX_INSPECT_PROJECT_ID_LENGTH",
    "MAX_PIPELINE_ERRORS",
    "MAX_PIPELINE_WARNINGS",
    "NEXT_RECOMMENDED_ACTIONS",
    "PHASE4_PIPELINE_STAGE_ORDER",
    "PHASE4_PROJECT_MEMORY_FLAG",
    "PIPELINE_STAGE_STATUSES",
    "PIPELINE_STATUSES",
    "Phase4PipelineHealth",
    "Phase4PipelineResult",
    "Phase4PipelineStageResult",
    "Phase4ProjectInspect",
    "Phase4ProjectInspectSummary",
    "Phase4ProjectMemoryPipelineResult",
    "get_phase4_pipeline_health",
    "get_phase4_project_inspect",
    "is_phase4_project_memory_enabled",
    "run_phase4_project_memory_pipeline",
]
