"""Deterministic, validated persistence for Phase 4 Project Memory.

Step 9 deliberately accepts already-built Phase 4 records.  It does not load
upstream artifacts, run extraction, expose APIs, or perform retrieval.  The
persisted identity excludes timestamps, destination paths, and transient
diagnostics; it includes every safe semantic record stored in each project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from backend.phase4_claim_boundary import (
    Phase4ClaimBoundaryReport,
    validate_phase4_claim_boundary,
)
from backend.phase4_models import (
    PHASE4_SCHEMA_VERSION,
    ClaimSubjectType,
    EvidenceStatus,
    Phase4CapabilityFact,
    Phase4ClaimBoundary,
    Phase4EvidenceFact,
    Phase4PipelineWarning,
    Phase4ProjectMemory,
    WarningSeverity,
    build_phase4_stable_id,
)


ROOT_DIR = Path(__file__).parents[1]
DEFAULT_PHASE4_PROJECT_MEMORY_PATH = ROOT_DIR / "information" / "project_memory_phase4.json"

MAX_PROJECTS = 1_000
MAX_EVIDENCE_FACTS = 100_000
MAX_CAPABILITY_FACTS = 10_000
MAX_CLAIM_BOUNDARIES = 100_000
MAX_SERIALIZED_SIZE = 25 * 1024 * 1024
MAX_WARNINGS = 100
MAX_VALIDATION_ERRORS = 100

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WARNING_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_FORBIDDEN_STORAGE_FIELDS = frozenset({
    "raw_text",
    "raw_content",
    "raw_patch",
    "full_patch",
    "patch",
    "complete_diff",
    "diff_text",
    "source_code",
    "full_file_content",
    "repository_dump",
    "github_raw",
    "environment",
    "api_key",
    "access_token",
    "secret",
    "password",
    "private_key",
    "authorization",
    "authorization_header",
})
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
_BEARER_SECRET_RE = re.compile(r"(?i)^\s*bearer\s+[a-z0-9._~+/=-]{12,}\s*$")
_KNOWN_TOKEN_RE = re.compile(r"^(?:gh[oprsu]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})$")
_FIXED_WARNING_MESSAGES = {
    "capability_facts_empty": "No emitted capability facts were available for this project.",
    "context_only_project": "This project memory contains only a validated project boundary.",
}


class Phase4ProjectMemoryIntegrityError(ValueError):
    """Raised when deterministic IDs conflict or references fail closed."""


@dataclass(frozen=True)
class Phase4ProjectMemoryDiagnostics:
    evidence_fact_count: int = 0
    capability_fact_count: int = 0
    claim_boundary_count: int = 0
    project_boundary_count: int = 0
    allowed_claim_count: int = 0
    forbidden_claim_count: int = 0
    claim_truncation_count: int = 0
    projects_with_truncation: int = 0
    claim_type_truncation_counts: dict[str, int] = field(default_factory=dict)
    weak_fact_blocked_count: int = 0
    rejected_fact_blocked_count: int = 0
    low_quality_blocked_count: int = 0
    contextual_only_restricted_count: int = 0
    unsupported_metric_blocked_count: int = 0
    unsupported_impact_blocked_count: int = 0
    unsupported_capability_blocked_count: int = 0
    project_mismatch_count: int = 0
    conflict_count: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in _DIAGNOSTIC_COUNT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.claim_type_truncation_counts, Mapping):
            raise TypeError("claim_type_truncation_counts must be an object")
        normalized_counts: dict[str, int] = {}
        for key, value in self.claim_type_truncation_counts.items():
            if not isinstance(key, str) or not _WARNING_CODE_RE.fullmatch(key):
                raise ValueError("claim_type_truncation_counts keys must be safe identifiers")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("claim_type_truncation_counts values must be non-negative integers")
            normalized_counts[key] = value
        object.__setattr__(self, "claim_type_truncation_counts", dict(sorted(normalized_counts.items())))
        if not isinstance(self.warnings, (list, tuple)):
            raise TypeError("warnings must be a sequence")
        warning_codes: set[str] = set()
        for warning in self.warnings:
            if not isinstance(warning, str) or not _WARNING_CODE_RE.fullmatch(warning):
                raise ValueError("diagnostic warnings must contain safe codes only")
            warning_codes.add(warning)
        object.__setattr__(self, "warnings", tuple(sorted(warning_codes)[:MAX_WARNINGS]))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase4ProjectMemoryDiagnostics":
        if not isinstance(payload, Mapping):
            raise TypeError("diagnostics must be an object")
        allowed = set(_DIAGNOSTIC_COUNT_FIELDS) | {"claim_type_truncation_counts", "warnings"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unknown diagnostics fields")
        values = dict(payload)
        values["warnings"] = tuple(values.get("warnings", ()))
        return cls(**values)

    @classmethod
    def from_claim_boundary_report(
        cls,
        report: Phase4ClaimBoundaryReport,
        *,
        projects_with_truncation: int = 0,
        claim_type_truncation_counts: Mapping[str, int] | None = None,
        warnings: Iterable[str] = (),
    ) -> "Phase4ProjectMemoryDiagnostics":
        required_report_fields = (
            "evidence_fact_count", "capability_fact_count", "evidence_boundaries_created",
            "capability_boundaries_created", "project_boundaries_created", "allowed_claim_count",
            "forbidden_claim_count", "truncated_claim_count", "weak_fact_blocked_count",
            "rejected_fact_blocked_count", "low_quality_blocked_count",
            "contextual_only_restricted_count", "unsupported_metric_blocked_count",
            "unsupported_impact_blocked_count", "unsupported_capability_blocked_count",
            "project_mismatch_count", "conflict_count",
        )
        if not isinstance(report, Phase4ClaimBoundaryReport) and not all(
            hasattr(report, name) for name in required_report_fields
        ):
            raise TypeError("report must be a Phase4ClaimBoundaryReport")
        warning_codes = set(warnings)
        if report.truncated_claim_count:
            warning_codes.add("claim_budget_truncated")
        if report.capability_fact_count == 0:
            warning_codes.add("capability_facts_empty")
        return cls(
            evidence_fact_count=report.evidence_fact_count,
            capability_fact_count=report.capability_fact_count,
            claim_boundary_count=(
                report.evidence_boundaries_created
                + report.capability_boundaries_created
                + report.project_boundaries_created
            ),
            project_boundary_count=report.project_boundaries_created,
            allowed_claim_count=report.allowed_claim_count,
            forbidden_claim_count=report.forbidden_claim_count,
            claim_truncation_count=report.truncated_claim_count,
            projects_with_truncation=projects_with_truncation,
            claim_type_truncation_counts=dict(claim_type_truncation_counts or {}),
            weak_fact_blocked_count=report.weak_fact_blocked_count,
            rejected_fact_blocked_count=report.rejected_fact_blocked_count,
            low_quality_blocked_count=report.low_quality_blocked_count,
            contextual_only_restricted_count=report.contextual_only_restricted_count,
            unsupported_metric_blocked_count=report.unsupported_metric_blocked_count,
            unsupported_impact_blocked_count=report.unsupported_impact_blocked_count,
            unsupported_capability_blocked_count=report.unsupported_capability_blocked_count,
            project_mismatch_count=report.project_mismatch_count,
            conflict_count=report.conflict_count,
            warnings=tuple(warning_codes),
        )


_DIAGNOSTIC_COUNT_FIELDS = (
    "evidence_fact_count",
    "capability_fact_count",
    "claim_boundary_count",
    "project_boundary_count",
    "allowed_claim_count",
    "forbidden_claim_count",
    "claim_truncation_count",
    "projects_with_truncation",
    "weak_fact_blocked_count",
    "rejected_fact_blocked_count",
    "low_quality_blocked_count",
    "contextual_only_restricted_count",
    "unsupported_metric_blocked_count",
    "unsupported_impact_blocked_count",
    "unsupported_capability_blocked_count",
    "project_mismatch_count",
    "conflict_count",
)


@dataclass(frozen=True)
class Phase4ProjectMemoryBuildReport:
    project_count: int
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int
    duplicate_evidence_fact_count: int = 0
    duplicate_capability_fact_count: int = 0
    duplicate_claim_boundary_count: int = 0
    not_present_capability_count: int = 0
    context_only_project_count: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase4ProjectMemorySnapshot:
    schema_version: str
    content_hash: str
    projects: tuple[Phase4ProjectMemory, ...]
    diagnostics: Phase4ProjectMemoryDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return _snapshot_payload(self, include_hash=True)


@dataclass(frozen=True)
class Phase4ProjectMemoryValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = ""
    content_hash: str = ""
    project_count: int = 0
    evidence_fact_count: int = 0
    capability_fact_count: int = 0
    claim_boundary_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase4ProjectMemoryPersistenceReport:
    status: str
    path: str
    schema_version: str
    content_hash: str
    project_count: int
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int
    allowed_claim_count: int
    forbidden_claim_count: int
    claim_truncation_count: int
    bytes_written: int
    previous_artifact_preserved: bool
    round_trip_validated: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase4ProjectMemoryLoadResult:
    status: str
    snapshot: Phase4ProjectMemorySnapshot | None
    validation: Phase4ProjectMemoryValidationReport
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "validation": self.validation.to_dict(),
            "warnings": list(self.warnings),
        }


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs).encode("utf-8")


def _payload_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalized_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _reject_prohibited_content(value: Any, *, field_name: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Phase4ProjectMemoryIntegrityError("non_string_storage_field")
            normalized = _normalized_field_name(key)
            if normalized in _FORBIDDEN_STORAGE_FIELDS:
                raise Phase4ProjectMemoryIntegrityError(f"prohibited_storage_field:{normalized}")
            _reject_prohibited_content(item, field_name=normalized)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_prohibited_content(item, field_name=field_name)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase4ProjectMemoryIntegrityError("non_finite_number")
    if isinstance(value, str):
        if _PRIVATE_KEY_RE.search(value):
            raise Phase4ProjectMemoryIntegrityError("private_key_value")
        if _BEARER_SECRET_RE.fullmatch(value) or _KNOWN_TOKEN_RE.fullmatch(value.strip()):
            raise Phase4ProjectMemoryIntegrityError("secret_like_value")
        return
    if value is not None and not isinstance(value, (bool, int)):
        raise Phase4ProjectMemoryIntegrityError("non_json_value")


def _is_absolute_source_path(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.replace("\\", "/")
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(normalized).is_absolute()
        or normalized.startswith(("~/", "//"))
    )


def _record_payload(record: Any) -> dict[str, Any]:
    payload = record.to_dict()
    _reject_prohibited_content(payload)
    return payload


def _dedupe_records(
    records: Sequence[Any],
    *,
    expected_type: type,
    id_field: str,
) -> tuple[list[Any], int]:
    by_id: dict[str, tuple[str, Any]] = {}
    duplicates = 0
    for record in records:
        if not isinstance(record, expected_type):
            raise TypeError(f"{id_field} records must be {expected_type.__name__} values")
        payload = _record_payload(record)
        serialized = _canonical_json_bytes(payload).decode("utf-8")
        record_id = getattr(record, id_field)
        previous = by_id.get(record_id)
        if previous is None:
            by_id[record_id] = (serialized, record)
        elif previous[0] == serialized:
            duplicates += 1
        else:
            raise Phase4ProjectMemoryIntegrityError(f"same_{id_field}_different_payload")
    return [by_id[key][1] for key in sorted(by_id)], duplicates


def _source_hash_summary(facts: Sequence[Phase4EvidenceFact]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        for ref in fact.source_refs:
            if _is_absolute_source_path(ref.file_path):
                raise Phase4ProjectMemoryIntegrityError("absolute_source_path")
            by_type.setdefault(ref.source_type, []).append({
                "source_id": ref.source_id,
                "content_hash": ref.content_hash,
                "repo": ref.repo,
                "commit_sha": ref.commit_sha,
                "file_path": ref.file_path,
                "symbol": ref.symbol,
                "start_line": ref.start_line,
                "end_line": ref.end_line,
            })
    result: dict[str, Any] = {}
    for source_type, refs in sorted(by_type.items()):
        canonical_refs = sorted(refs, key=lambda item: _canonical_json_bytes(item))
        result[source_type] = {
            "reference_count": len(canonical_refs),
            "content_hash": _payload_hash(canonical_refs),
        }
    return result


def _quality_summary(facts: Sequence[Phase4EvidenceFact]) -> dict[str, int]:
    return {
        "accepted_count": sum(fact.status is EvidenceStatus.ACCEPTED for fact in facts),
        "supporting_count": sum(fact.status is EvidenceStatus.SUPPORTING for fact in facts),
        "weak_count": sum(fact.status is EvidenceStatus.WEAK for fact in facts),
        "rejected_count": sum(fact.status is EvidenceStatus.REJECTED for fact in facts),
    }


def _expected_project_memory_id(memory: Phase4ProjectMemory) -> str:
    return build_phase4_stable_id("p4pm_", memory.project_id, {
        "schema_version": memory.schema_version,
        "project_id": memory.project_id,
        "project_name": memory.project_name,
        "source_hashes": memory.source_hashes,
        "evidence_facts": memory.evidence_facts,
        "capability_facts": memory.capability_facts,
        "claim_boundaries": memory.claim_boundaries,
        "quality_summary": memory.quality_summary,
    })


def _project_subject_id(project_id: str) -> str:
    if len(project_id) <= 100:
        return project_id
    return build_phase4_stable_id("p4claim_", project_id, {"subject_type": "project"})


def _validate_project_cross_references(memory: Phase4ProjectMemory) -> tuple[str, ...]:
    errors: set[str] = set()
    evidence_by_id = {fact.evidence_fact_id: fact for fact in memory.evidence_facts}
    capability_by_id = {fact.capability_id: fact for fact in memory.capability_facts}
    for fact in memory.evidence_facts:
        if fact.project_id != memory.project_id:
            errors.add("evidence_project_mismatch")
        if any(ref.project_id != memory.project_id for ref in fact.source_refs):
            errors.add("source_reference_project_mismatch")
        if any(_is_absolute_source_path(ref.file_path) for ref in fact.source_refs):
            errors.add("absolute_source_path")
    for capability in memory.capability_facts:
        if not capability.present:
            errors.add("capability_not_present")
        if capability.project_id != memory.project_id:
            errors.add("capability_project_mismatch")
        for evidence_id in capability.source_evidence_fact_ids:
            fact = evidence_by_id.get(evidence_id)
            if fact is None:
                errors.add("unknown_capability_evidence_fact_id")
            elif fact.project_id != memory.project_id:
                errors.add("cross_project_capability_evidence")
    for boundary in memory.claim_boundaries:
        if boundary.project_id != memory.project_id:
            errors.add("boundary_project_mismatch")
        if boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT:
            fact = evidence_by_id.get(boundary.subject_id)
            if fact is None:
                errors.add("unknown_evidence_boundary_subject")
            elif fact.project_id != memory.project_id:
                errors.add("cross_project_evidence_boundary")
        elif boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT:
            capability = capability_by_id.get(boundary.subject_id)
            if capability is None:
                errors.add("unknown_capability_boundary_subject")
            elif capability.project_id != memory.project_id:
                errors.add("cross_project_capability_boundary")
        elif boundary.subject_type is ClaimSubjectType.PROJECT:
            if boundary.subject_id != _project_subject_id(memory.project_id):
                errors.add("project_boundary_subject_mismatch")
        result = validate_phase4_claim_boundary(
            boundary,
            evidence_facts_by_id=evidence_by_id,
            capability_facts_by_id=capability_by_id,
        )
        errors.update(result.errors)
    return tuple(sorted(errors))


def build_phase4_project_memories(
    evidence_facts: Iterable[Phase4EvidenceFact],
    capability_facts: Iterable[Phase4CapabilityFact],
    claim_boundaries: Iterable[Phase4ClaimBoundary],
    *,
    diagnostics: Phase4ProjectMemoryDiagnostics | None = None,
) -> tuple[list[Phase4ProjectMemory], Phase4ProjectMemoryBuildReport]:
    """Group validated records by exact project ID without file access."""

    evidence_input = list(evidence_facts)
    capability_input = list(capability_facts)
    boundary_input = list(claim_boundaries)
    evidence, duplicate_evidence = _dedupe_records(
        evidence_input, expected_type=Phase4EvidenceFact, id_field="evidence_fact_id"
    )
    capabilities, duplicate_capabilities = _dedupe_records(
        capability_input, expected_type=Phase4CapabilityFact, id_field="capability_id"
    )
    boundaries, duplicate_boundaries = _dedupe_records(
        boundary_input, expected_type=Phase4ClaimBoundary, id_field="boundary_id"
    )
    not_present = sum(not capability.present for capability in capabilities)
    capabilities = [capability for capability in capabilities if capability.present]
    project_ids = sorted({
        item.project_id for item in [*evidence, *capabilities, *boundaries]
    })
    if len(project_ids) > MAX_PROJECTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_project_count_exceeded")
    if len(evidence) > MAX_EVIDENCE_FACTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_evidence_fact_count_exceeded")
    if len(capabilities) > MAX_CAPABILITY_FACTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_capability_fact_count_exceeded")
    if len(boundaries) > MAX_CLAIM_BOUNDARIES:
        raise Phase4ProjectMemoryIntegrityError("maximum_claim_boundary_count_exceeded")
    memories: list[Phase4ProjectMemory] = []
    context_only = 0
    for project_id in project_ids:
        project_evidence = [fact for fact in evidence if fact.project_id == project_id]
        project_capabilities = [fact for fact in capabilities if fact.project_id == project_id]
        project_boundaries = [boundary for boundary in boundaries if boundary.project_id == project_id]
        warning_codes: list[str] = []
        if not project_capabilities:
            warning_codes.append("capability_facts_empty")
        if not project_evidence and not project_capabilities:
            context_only += 1
            warning_codes.append("context_only_project")
        warnings = [
            Phase4PipelineWarning(
                code=code,
                message=_FIXED_WARNING_MESSAGES[code],
                project_id=project_id,
                severity=WarningSeverity.WARNING,
            )
            for code in sorted(set(warning_codes))
        ]
        memory = Phase4ProjectMemory(
            project_id=project_id,
            project_name=project_id,
            source_hashes=_source_hash_summary(project_evidence),
            evidence_facts=project_evidence,
            capability_facts=sorted(
                project_capabilities, key=lambda item: (item.capability_type, item.capability_id)
            ),
            claim_boundaries=sorted(project_boundaries, key=lambda item: (
                item.subject_type.value, item.subject_id, item.boundary_id,
            )),
            quality_summary=_quality_summary(project_evidence),
            warnings=warnings,
        )
        cross_reference_errors = _validate_project_cross_references(memory)
        if cross_reference_errors:
            raise Phase4ProjectMemoryIntegrityError(";".join(cross_reference_errors))
        _reject_prohibited_content(memory.to_dict())
        memories.append(memory)
    memories.sort(key=lambda item: (item.project_id, item.project_memory_id))
    report_warnings = set(diagnostics.warnings if diagnostics is not None else ())
    if not capabilities:
        report_warnings.add("capability_facts_empty")
    report = Phase4ProjectMemoryBuildReport(
        project_count=len(memories),
        evidence_fact_count=len(evidence),
        capability_fact_count=len(capabilities),
        claim_boundary_count=len(boundaries),
        duplicate_evidence_fact_count=duplicate_evidence,
        duplicate_capability_fact_count=duplicate_capabilities,
        duplicate_claim_boundary_count=duplicate_boundaries,
        not_present_capability_count=not_present,
        context_only_project_count=context_only,
        warnings=tuple(sorted(report_warnings)[:MAX_WARNINGS]),
    )
    return memories, report


def _snapshot_payload(
    snapshot: Phase4ProjectMemorySnapshot,
    *,
    include_hash: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": snapshot.schema_version,
        "project_count": len(snapshot.projects),
        "projects": [project.to_dict() for project in snapshot.projects],
        "diagnostics": snapshot.diagnostics.to_dict(),
    }
    if include_hash:
        payload["content_hash"] = snapshot.content_hash
    return payload


def _snapshot_content_hash(snapshot: Phase4ProjectMemorySnapshot) -> str:
    return _payload_hash(_snapshot_payload(snapshot, include_hash=False))


def _snapshot_counts(projects: Sequence[Phase4ProjectMemory]) -> tuple[int, int, int, int]:
    return (
        len(projects),
        sum(len(project.evidence_facts) for project in projects),
        sum(len(project.capability_facts) for project in projects),
        sum(len(project.claim_boundaries) for project in projects),
    )


def build_phase4_project_memory_snapshot(
    project_memories: Iterable[Phase4ProjectMemory],
    *,
    diagnostics: Phase4ProjectMemoryDiagnostics | None = None,
) -> Phase4ProjectMemorySnapshot:
    """Construct and hash a canonical immutable snapshot without writing it."""

    projects = list(project_memories)
    if any(not isinstance(project, Phase4ProjectMemory) for project in projects):
        raise TypeError("project_memories must contain Phase4ProjectMemory values")
    by_id: dict[str, tuple[str, Phase4ProjectMemory]] = {}
    project_ids: set[str] = set()
    for project in projects:
        payload = _record_payload(project)
        serialized = _canonical_json_bytes(payload).decode("utf-8")
        previous = by_id.get(project.project_memory_id)
        if previous is not None and previous[0] != serialized:
            raise Phase4ProjectMemoryIntegrityError("same_project_memory_id_different_payload")
        if project.project_id in project_ids:
            raise Phase4ProjectMemoryIntegrityError("duplicate_project_id")
        project_ids.add(project.project_id)
        by_id[project.project_memory_id] = (serialized, project)
    if any(project.project_memory_id != _expected_project_memory_id(project) for project in projects):
        raise Phase4ProjectMemoryIntegrityError("invalid_project_memory_id")
    ordered = tuple(sorted((item[1] for item in by_id.values()), key=lambda item: (
        item.project_id, item.project_memory_id,
    )))
    project_count, evidence_count, capability_count, boundary_count = _snapshot_counts(ordered)
    if project_count > MAX_PROJECTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_project_count_exceeded")
    if evidence_count > MAX_EVIDENCE_FACTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_evidence_fact_count_exceeded")
    if capability_count > MAX_CAPABILITY_FACTS:
        raise Phase4ProjectMemoryIntegrityError("maximum_capability_fact_count_exceeded")
    if boundary_count > MAX_CLAIM_BOUNDARIES:
        raise Phase4ProjectMemoryIntegrityError("maximum_claim_boundary_count_exceeded")
    project_boundary_count = sum(
        boundary.subject_type is ClaimSubjectType.PROJECT
        for project in ordered for boundary in project.claim_boundaries
    )
    base = diagnostics or Phase4ProjectMemoryDiagnostics(
        allowed_claim_count=sum(
            len(boundary.allowed_claims) for project in ordered for boundary in project.claim_boundaries
        ),
        forbidden_claim_count=sum(
            len(boundary.forbidden_claims) for project in ordered for boundary in project.claim_boundaries
        ),
    )
    normalized_diagnostics = replace(
        base,
        evidence_fact_count=evidence_count,
        capability_fact_count=capability_count,
        claim_boundary_count=boundary_count,
        project_boundary_count=project_boundary_count,
    )
    snapshot = Phase4ProjectMemorySnapshot(
        schema_version=PHASE4_SCHEMA_VERSION,
        content_hash="",
        projects=ordered,
        diagnostics=normalized_diagnostics,
    )
    snapshot = replace(snapshot, content_hash=_snapshot_content_hash(snapshot))
    validation = validate_phase4_project_memory_snapshot(snapshot)
    if not validation.valid:
        raise Phase4ProjectMemoryIntegrityError(validation.errors[0])
    return snapshot


def _bounded_errors(errors: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(errors))[:MAX_VALIDATION_ERRORS])


def validate_phase4_project_memory_snapshot(
    snapshot: Phase4ProjectMemorySnapshot,
) -> Phase4ProjectMemoryValidationReport:
    """Validate schema, identities, references, limits, safety, and hash."""

    errors: set[str] = set()
    if not isinstance(snapshot, Phase4ProjectMemorySnapshot):
        return Phase4ProjectMemoryValidationReport(valid=False, errors=("invalid_snapshot_type",))
    if snapshot.schema_version != PHASE4_SCHEMA_VERSION:
        errors.add("unsupported_schema_version" if snapshot.schema_version else "missing_schema_version")
    if not isinstance(snapshot.diagnostics, Phase4ProjectMemoryDiagnostics):
        errors.add("invalid_diagnostics")
        diagnostics = Phase4ProjectMemoryDiagnostics()
    else:
        diagnostics = snapshot.diagnostics
    projects = snapshot.projects
    if not isinstance(projects, tuple):
        errors.add("projects_must_be_tuple")
        projects = tuple(projects) if isinstance(projects, (list, tuple)) else ()
    if any(not isinstance(project, Phase4ProjectMemory) for project in projects):
        errors.add("invalid_project_memory_type")
        projects = tuple(project for project in projects if isinstance(project, Phase4ProjectMemory))
    ordered = tuple(sorted(projects, key=lambda item: (item.project_id, item.project_memory_id)))
    if tuple(projects) != ordered:
        errors.add("non_deterministic_project_order")
    project_count, evidence_count, capability_count, boundary_count = _snapshot_counts(ordered)
    if project_count > MAX_PROJECTS:
        errors.add("maximum_project_count_exceeded")
    if evidence_count > MAX_EVIDENCE_FACTS:
        errors.add("maximum_evidence_fact_count_exceeded")
    if capability_count > MAX_CAPABILITY_FACTS:
        errors.add("maximum_capability_fact_count_exceeded")
    if boundary_count > MAX_CLAIM_BOUNDARIES:
        errors.add("maximum_claim_boundary_count_exceeded")
    project_ids: set[str] = set()
    memory_ids: dict[str, str] = {}
    evidence_ids: dict[str, str] = {}
    capability_ids: dict[str, str] = {}
    boundary_ids: dict[str, str] = {}
    for project in ordered:
        try:
            payload = _record_payload(project)
            Phase4ProjectMemory.from_dict(payload)
        except (TypeError, ValueError, Phase4ProjectMemoryIntegrityError):
            errors.add("invalid_project_memory")
            continue
        if project.project_id in project_ids:
            errors.add("duplicate_project_id")
        project_ids.add(project.project_id)
        serialized = _canonical_json_bytes(payload).decode("utf-8")
        previous_memory = memory_ids.get(project.project_memory_id)
        if previous_memory is not None and previous_memory != serialized:
            errors.add("conflicting_project_memory_id")
        memory_ids[project.project_memory_id] = serialized
        if project.project_memory_id != _expected_project_memory_id(project):
            errors.add("invalid_project_memory_id")
        for record, record_id, seen, conflict_code in (
            *((fact, fact.evidence_fact_id, evidence_ids, "conflicting_evidence_fact_id") for fact in project.evidence_facts),
            *((fact, fact.capability_id, capability_ids, "conflicting_capability_id") for fact in project.capability_facts),
            *((boundary, boundary.boundary_id, boundary_ids, "conflicting_boundary_id") for boundary in project.claim_boundaries),
        ):
            current = _canonical_json_bytes(record.to_dict()).decode("utf-8")
            previous = seen.get(record_id)
            if previous is not None and previous != current:
                errors.add(conflict_code)
            elif previous is not None:
                errors.add("duplicate_record_id")
            seen[record_id] = current
        errors.update(_validate_project_cross_references(project))
    if diagnostics.evidence_fact_count != evidence_count:
        errors.add("diagnostic_evidence_count_mismatch")
    if diagnostics.capability_fact_count != capability_count:
        errors.add("diagnostic_capability_count_mismatch")
    if diagnostics.claim_boundary_count != boundary_count:
        errors.add("diagnostic_boundary_count_mismatch")
    actual_project_boundaries = sum(
        boundary.subject_type is ClaimSubjectType.PROJECT
        for project in ordered for boundary in project.claim_boundaries
    )
    if diagnostics.project_boundary_count != actual_project_boundaries:
        errors.add("diagnostic_project_boundary_count_mismatch")
    try:
        payload = _snapshot_payload(snapshot, include_hash=True)
        _reject_prohibited_content(payload)
        serialized_size = len(_canonical_json_bytes(payload, pretty=True)) + 1
        if serialized_size > MAX_SERIALIZED_SIZE:
            errors.add("maximum_serialized_size_exceeded")
    except (TypeError, ValueError, Phase4ProjectMemoryIntegrityError):
        errors.add("unsafe_snapshot_content")
    if not isinstance(snapshot.content_hash, str) or not _SHA256_RE.fullmatch(snapshot.content_hash):
        errors.add("invalid_content_hash")
    else:
        try:
            if snapshot.content_hash != _snapshot_content_hash(snapshot):
                errors.add("content_hash_mismatch")
        except (TypeError, ValueError, Phase4ProjectMemoryIntegrityError):
            errors.add("content_hash_unavailable")
    final_errors = _bounded_errors(errors)
    return Phase4ProjectMemoryValidationReport(
        valid=not final_errors,
        errors=final_errors,
        warnings=tuple(diagnostics.warnings[:MAX_WARNINGS]),
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        project_count=project_count,
        evidence_fact_count=evidence_count,
        capability_fact_count=capability_count,
        claim_boundary_count=boundary_count,
    )


def serialize_phase4_project_memory_snapshot(
    snapshot: Phase4ProjectMemorySnapshot,
) -> bytes:
    """Return canonical UTF-8 JSON with exactly one trailing newline."""

    validation = validate_phase4_project_memory_snapshot(snapshot)
    if not validation.valid:
        raise Phase4ProjectMemoryIntegrityError(f"invalid_snapshot:{validation.errors[0]}")
    payload = _snapshot_payload(snapshot, include_hash=True)
    _reject_prohibited_content(payload)
    serialized = _canonical_json_bytes(payload, pretty=True) + b"\n"
    if len(serialized) > MAX_SERIALIZED_SIZE:
        raise Phase4ProjectMemoryIntegrityError("maximum_serialized_size_exceeded")
    return serialized


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate_json_key")
        result[key] = value
    return result


def _empty_validation(*errors: str) -> Phase4ProjectMemoryValidationReport:
    return Phase4ProjectMemoryValidationReport(valid=False, errors=_bounded_errors(errors))


def load_phase4_project_memory(
    path: str | Path | None = None,
) -> Phase4ProjectMemoryLoadResult:
    """Safely load one complete snapshot; never return partial project data."""

    destination = Path(path) if path is not None else DEFAULT_PHASE4_PROJECT_MEMORY_PATH
    if not destination.exists():
        return Phase4ProjectMemoryLoadResult(
            status="missing", snapshot=None, validation=_empty_validation("artifact_missing"),
            warnings=("artifact_missing",),
        )
    if destination.is_dir():
        return Phase4ProjectMemoryLoadResult(
            status="error", snapshot=None, validation=_empty_validation("destination_is_directory"),
            warnings=("destination_is_directory",),
        )
    try:
        if destination.stat().st_size > MAX_SERIALIZED_SIZE:
            return Phase4ProjectMemoryLoadResult(
                status="invalid", snapshot=None,
                validation=_empty_validation("maximum_serialized_size_exceeded"),
                warnings=("maximum_serialized_size_exceeded",),
            )
        raw = destination.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except Exception:
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("malformed_artifact"),
            warnings=("malformed_artifact",),
        )
    if not isinstance(payload, dict):
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("invalid_snapshot_shape"),
            warnings=("invalid_snapshot_shape",),
        )
    schema_version = payload.get("schema_version")
    if schema_version is None:
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("missing_schema_version"),
            warnings=("missing_schema_version",),
        )
    if schema_version != PHASE4_SCHEMA_VERSION:
        return Phase4ProjectMemoryLoadResult(
            status="unsupported_version", snapshot=None,
            validation=_empty_validation("unsupported_schema_version"),
            warnings=("unsupported_schema_version",),
        )
    required = {"schema_version", "content_hash", "project_count", "projects", "diagnostics"}
    if set(payload) != required:
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("invalid_snapshot_fields"),
            warnings=("invalid_snapshot_fields",),
        )
    if isinstance(payload.get("project_count"), bool) or not isinstance(payload.get("project_count"), int):
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("invalid_project_count"),
            warnings=("invalid_project_count",),
        )
    if not isinstance(payload.get("projects"), list) or payload["project_count"] != len(payload["projects"]):
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("project_count_mismatch"),
            warnings=("project_count_mismatch",),
        )
    try:
        for project in payload["projects"]:
            if not isinstance(project, Mapping) or "project_memory_id" not in project:
                raise ValueError("missing project memory ID")
        projects = tuple(Phase4ProjectMemory.from_dict(item) for item in payload["projects"])
        diagnostics = Phase4ProjectMemoryDiagnostics.from_dict(payload["diagnostics"])
        snapshot = Phase4ProjectMemorySnapshot(
            schema_version=schema_version,
            content_hash=payload.get("content_hash"),
            projects=projects,
            diagnostics=diagnostics,
        )
    except (TypeError, ValueError, Phase4ProjectMemoryIntegrityError):
        return Phase4ProjectMemoryLoadResult(
            status="invalid", snapshot=None, validation=_empty_validation("invalid_snapshot_payload"),
            warnings=("invalid_snapshot_payload",),
        )
    validation = validate_phase4_project_memory_snapshot(snapshot)
    if not validation.valid:
        status = "hash_mismatch" if any(
            code in validation.errors for code in ("content_hash_mismatch", "invalid_content_hash")
        ) else "invalid"
        return Phase4ProjectMemoryLoadResult(
            status=status, snapshot=None, validation=validation, warnings=validation.errors,
        )
    return Phase4ProjectMemoryLoadResult(
        status="ready", snapshot=snapshot, validation=validation,
        warnings=tuple(diagnostics.warnings),
    )


def _persistence_report(
    snapshot: Phase4ProjectMemorySnapshot,
    destination: Path,
    *,
    status: str,
    bytes_written: int = 0,
    previous_artifact_preserved: bool = False,
    round_trip_validated: bool = False,
    warnings: Iterable[str] = (),
) -> Phase4ProjectMemoryPersistenceReport:
    project_count, evidence_count, capability_count, boundary_count = _snapshot_counts(snapshot.projects)
    return Phase4ProjectMemoryPersistenceReport(
        status=status,
        path=str(destination),
        schema_version=snapshot.schema_version,
        content_hash=snapshot.content_hash,
        project_count=project_count,
        evidence_fact_count=evidence_count,
        capability_fact_count=capability_count,
        claim_boundary_count=boundary_count,
        allowed_claim_count=snapshot.diagnostics.allowed_claim_count,
        forbidden_claim_count=snapshot.diagnostics.forbidden_claim_count,
        claim_truncation_count=snapshot.diagnostics.claim_truncation_count,
        bytes_written=bytes_written,
        previous_artifact_preserved=previous_artifact_preserved,
        round_trip_validated=round_trip_validated,
        warnings=tuple(sorted(set(warnings))[:MAX_WARNINGS]),
    )


def _write_temp_bytes(destination: Path, serialized: bytes) -> Path:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise


def _sync_parent_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        os.fsync(descriptor)
    except Exception:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _restore_previous_artifact(destination: Path, previous_bytes: bytes | None) -> bool:
    if previous_bytes is None:
        try:
            destination.unlink(missing_ok=True)
            return not destination.exists()
        except Exception:
            return False
    temp_path: Path | None = None
    try:
        temp_path = _write_temp_bytes(destination, previous_bytes)
        os.replace(temp_path, destination)
        temp_path = None
        return destination.read_bytes() == previous_bytes
    except Exception:
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def persist_phase4_project_memory(
    snapshot: Phase4ProjectMemorySnapshot,
    path: str | Path | None = None,
    *,
    replace_invalid: bool = False,
) -> Phase4ProjectMemoryPersistenceReport:
    """Atomically persist a validated snapshot with unchanged-write detection."""

    destination = Path(path) if path is not None else DEFAULT_PHASE4_PROJECT_MEMORY_PATH
    validation = validate_phase4_project_memory_snapshot(snapshot)
    if not validation.valid:
        return _persistence_report(
            snapshot, destination, status="failed", previous_artifact_preserved=destination.exists(),
            warnings=("invalid_snapshot",),
        )
    try:
        serialized = serialize_phase4_project_memory_snapshot(snapshot)
    except (TypeError, ValueError, Phase4ProjectMemoryIntegrityError):
        return _persistence_report(
            snapshot, destination, status="failed", previous_artifact_preserved=destination.exists(),
            warnings=("serialization_failed",),
        )
    if destination.exists() and destination.is_dir():
        return _persistence_report(
            snapshot, destination, status="failed", previous_artifact_preserved=True,
            warnings=("destination_is_directory",),
        )
    previous_bytes: bytes | None = None
    existed = destination.exists()
    if existed:
        existing = load_phase4_project_memory(destination)
        if existing.status == "ready" and existing.snapshot is not None:
            if existing.snapshot.content_hash == snapshot.content_hash:
                return _persistence_report(
                    snapshot, destination, status="unchanged", previous_artifact_preserved=True,
                    round_trip_validated=True,
                )
            try:
                previous_bytes = destination.read_bytes()
            except Exception:
                return _persistence_report(
                    snapshot, destination, status="failed", previous_artifact_preserved=True,
                    warnings=("existing_artifact_read_failed",),
                )
        elif not replace_invalid:
            return _persistence_report(
                snapshot, destination, status="failed", previous_artifact_preserved=True,
                warnings=("invalid_existing_artifact",),
            )
        else:
            try:
                previous_bytes = destination.read_bytes()
            except Exception:
                previous_bytes = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return _persistence_report(
            snapshot, destination, status="failed", previous_artifact_preserved=existed,
            warnings=("parent_directory_create_failed",),
        )
    temp_path: Path | None = None
    try:
        temp_path = _write_temp_bytes(destination, serialized)
        temp_load = load_phase4_project_memory(temp_path)
        if temp_load.status != "ready" or temp_load.snapshot != snapshot:
            return _persistence_report(
                snapshot, destination, status="failed", previous_artifact_preserved=existed,
                warnings=("temporary_round_trip_failed",),
            )
        try:
            os.replace(temp_path, destination)
        except Exception:
            return _persistence_report(
                snapshot, destination, status="failed", previous_artifact_preserved=existed,
                warnings=("atomic_replace_failed",),
            )
        temp_path = None
        _sync_parent_directory(destination.parent)
        loaded = load_phase4_project_memory(destination)
        if loaded.status != "ready" or loaded.snapshot != snapshot:
            preserved = _restore_previous_artifact(destination, previous_bytes)
            return _persistence_report(
                snapshot, destination, status="failed", previous_artifact_preserved=preserved,
                warnings=("round_trip_failed",),
            )
        return _persistence_report(
            snapshot,
            destination,
            status="updated" if existed else "created",
            bytes_written=len(serialized),
            previous_artifact_preserved=False,
            round_trip_validated=True,
        )
    except Exception:
        return _persistence_report(
            snapshot, destination, status="failed", previous_artifact_preserved=existed,
            warnings=("write_failed",),
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_and_persist_phase4_project_memory(
    evidence_facts: Iterable[Phase4EvidenceFact],
    capability_facts: Iterable[Phase4CapabilityFact],
    claim_boundaries: Iterable[Phase4ClaimBoundary],
    *,
    diagnostics: Phase4ProjectMemoryDiagnostics | None = None,
    path: str | Path | None = None,
    replace_invalid: bool = False,
) -> Phase4ProjectMemoryPersistenceReport:
    """Convenience wrapper over supplied records; it is not pipeline orchestration."""

    memories, _report = build_phase4_project_memories(
        evidence_facts, capability_facts, claim_boundaries, diagnostics=diagnostics
    )
    snapshot = build_phase4_project_memory_snapshot(memories, diagnostics=diagnostics)
    return persist_phase4_project_memory(
        snapshot, path=path, replace_invalid=replace_invalid
    )


__all__ = [
    "DEFAULT_PHASE4_PROJECT_MEMORY_PATH",
    "MAX_CAPABILITY_FACTS",
    "MAX_CLAIM_BOUNDARIES",
    "MAX_EVIDENCE_FACTS",
    "MAX_PROJECTS",
    "MAX_SERIALIZED_SIZE",
    "MAX_WARNINGS",
    "Phase4ProjectMemoryBuildReport",
    "Phase4ProjectMemoryDiagnostics",
    "Phase4ProjectMemoryIntegrityError",
    "Phase4ProjectMemoryLoadResult",
    "Phase4ProjectMemoryPersistenceReport",
    "Phase4ProjectMemorySnapshot",
    "Phase4ProjectMemoryValidationReport",
    "build_and_persist_phase4_project_memory",
    "build_phase4_project_memories",
    "build_phase4_project_memory_snapshot",
    "load_phase4_project_memory",
    "persist_phase4_project_memory",
    "serialize_phase4_project_memory_snapshot",
    "validate_phase4_project_memory_snapshot",
]
