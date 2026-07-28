"""Strict, deterministic project evidence and capability memory schemas.

This module contains data shapes and pure normalization/serialization helpers
only.  It performs no extraction, persistence, retrieval, or generation.

Oversized bounded content is rejected. Unknown fields supplied to ``from_dict``
are rejected, and forbidden raw-content/secret keys are rejected recursively.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
import json
import re
import types
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints


PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION = "project_evidence_memory.v1"
MAX_TITLE_LENGTH = 300
MAX_SUMMARY_LENGTH = 2_000
MAX_SIGNAL_LENGTH = 1_000
MAX_DETAIL_LENGTH = 2_000
MAX_CLAIM_LENGTH = 1_000
MAX_TAG_LENGTH = 100
MAX_WARNING_MESSAGE_LENGTH = 1_000
MAX_NOTES_LENGTH = 1_000
MAX_LIST_ITEMS = 200
MAX_METADATA_ITEMS = 100
MAX_METADATA_DEPTH = 6
MAX_METADATA_STRING_LENGTH = 2_000

FORBIDDEN_CONTENT_KEYS = frozenset({
    "raw_text", "raw_patch", "full_patch", "complete_diff", "repository_dump",
    "full_file_content", "access_token", "api_key", "secret", "password",
    "authorization_header",
})
ALLOWED_ID_PREFIXES = frozenset({"esr_", "pei_", "pef_", "pcf_", "pcb_", "pem_"})
ORDER_INSENSITIVE_FIELDS = frozenset({
    "technical_tags",
    "source_evidence_fact_ids",
    "project_ids",
    "allowed_claims",
    "allowed_resume_claims",
    "forbidden_claims",
})
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T", bound="ProjectEvidenceModel")


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MetricSupport(str, Enum):
    NONE = "none"
    APPROXIMATE = "approximate"
    EXPLICIT = "explicit"


class EvidenceStatus(str, Enum):
    ACCEPTED = "accepted"
    SUPPORTING = "supporting"
    WEAK = "weak"
    REJECTED = "rejected"


class PipelineStatus(str, Enum):
    DISABLED = "disabled"
    EMPTY = "empty"
    COMPLETE = "complete"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    ERROR = "error"


class EvidenceType(str, Enum):
    FEATURE = "feature"
    BUG_FIX = "bug_fix"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"
    VALIDATION = "validation"
    FAILURE_RECOVERY = "failure_recovery"
    DATA_PERSISTENCE = "data_persistence"
    RETRIEVAL = "retrieval"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"
    TESTING = "testing"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class ClaimSubjectType(str, Enum):
    EVIDENCE_FACT = "evidence_fact"
    CAPABILITY_FACT = "capability_fact"
    PROJECT = "project"


class WarningSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


def _bounded(value: str, name: str, maximum: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = " ".join(value.split())
    if required and not value:
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    return value


def _identifier(value: str, name: str) -> str:
    value = _bounded(value, name, 300, required=True).lower()
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must be a normalized identifier")
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from exc


def _strings(values: list[str], name: str, maximum: int, *, sort: bool = False) -> list[str]:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    if len(values) > MAX_LIST_ITEMS:
        raise ValueError(f"{name} exceeds maximum item count {MAX_LIST_ITEMS}")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        normalized = _bounded(item, name, maximum, required=True)
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return sorted(result, key=lambda item: (item.casefold(), item)) if sort else result


def _json_safe(value: Any, *, path: str = "value", depth: int = 0) -> Any:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"{path} exceeds maximum nesting depth {MAX_METADATA_DEPTH}")
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, str):
        return _bounded(value, path, MAX_METADATA_STRING_LENGTH)
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_ITEMS:
            raise ValueError(f"{path} exceeds maximum item count {MAX_METADATA_ITEMS}")
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized_key = key.strip()
            if normalized_key.casefold() in FORBIDDEN_CONTENT_KEYS:
                raise ValueError(f"forbidden project evidence field: {normalized_key}")
            output[normalized_key] = _json_safe(item, path=f"{path}.{normalized_key}", depth=depth + 1)
        return {key: output[key] for key in sorted(output)}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_ITEMS:
            raise ValueError(f"{path} exceeds maximum item count {MAX_LIST_ITEMS}")
        return [_json_safe(item, path=f"{path}[]", depth=depth + 1) for item in value]
    raise TypeError(f"{path} contains non-JSON-safe value {type(value).__name__}")


def _canonical(value: Any, *, field_name: str | None = None) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ProjectEvidenceModel):
        return _canonical(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(value[key], field_name=str(key))
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        items = [_canonical(item) for item in value]
        if field_name in ORDER_INSENSITIVE_FIELDS:
            return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return items
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if isinstance(value, str):
        return " ".join(value.split())
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"unsupported stable-ID value: {type(value).__name__}")


def build_project_evidence_stable_id(prefix: str, project_id: str, normalized_payload: object) -> str:
    if prefix not in ALLOWED_ID_PREFIXES:
        raise ValueError(f"unsupported project evidence ID prefix: {prefix!r}")
    project = _bounded(project_id, "project_id", 300, required=True)
    canonical = _canonical({"project_id": project, "payload": normalized_payload})
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _decode(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError("expected a list")
        return [_decode(args[0], item) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("expected an object")
        return dict(value)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [item for item in args if item is not type(None)]
        return _decode(choices[0], value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, ProjectEvidenceModel):
        if not isinstance(value, dict):
            raise TypeError("expected an object")
        return annotation.from_dict(value)
    return value


class ProjectEvidenceModel:
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            result[item.name] = _serialize(getattr(self, item.name))
        return result

    @classmethod
    def from_dict(cls: type[_T], payload: Mapping[str, Any]) -> _T:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects an object")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
        hints = get_type_hints(cls)
        return cls(**{key: _decode(hints[key], value) for key, value in payload.items()})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ProjectEvidenceModel):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(value[key]) for key in sorted(value)}
    return value


@dataclass(frozen=True)
class EvidenceSourceRef(ProjectEvidenceModel):
    source_type: str
    source_id: str
    project_id: str
    content_hash: str
    repo: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_type", "source_id", "project_id"):
            object.__setattr__(self, name, _bounded(getattr(self, name), name, 300, required=True))
        digest = _bounded(self.content_hash, "content_hash", 64, required=True).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("content_hash must be a 64-character SHA-256 hex digest")
        object.__setattr__(self, "content_hash", digest)
        if self.file_path is not None:
            object.__setattr__(self, "file_path", _bounded(self.file_path.replace("\\", "/"), "file_path", 500))
        for name in ("repo", "commit_sha", "symbol"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _bounded(value, name, 500))
        if self.start_line is not None and (isinstance(self.start_line, bool) or self.start_line <= 0):
            raise ValueError("start_line must be a positive integer")
        if self.end_line is not None and (isinstance(self.end_line, bool) or self.end_line <= 0):
            raise ValueError("end_line must be a positive integer")
        if self.start_line is not None and self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must not be lower than start_line")
        object.__setattr__(self, "metadata", _json_safe(self.metadata, path="metadata"))


@dataclass(frozen=True)
class ProjectEvidenceInput(ProjectEvidenceModel):
    project_id: str
    input_type: str
    title: str
    summary: str
    source_refs: list[EvidenceSourceRef]
    content_hash: str
    input_id: str = ""
    problem_signal: str | None = None
    mechanism_signals: list[str] = field(default_factory=list)
    implementation_signals: list[str] = field(default_factory=list)
    impact_signals: list[str] = field(default_factory=list)
    technical_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        project = _bounded(self.project_id, "project_id", 300, required=True)
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "input_type", _bounded(self.input_type, "input_type", 100, required=True))
        object.__setattr__(self, "title", _bounded(self.title, "title", MAX_TITLE_LENGTH, required=True))
        object.__setattr__(self, "summary", _bounded(self.summary, "summary", MAX_SUMMARY_LENGTH, required=True))
        if not self.source_refs:
            raise ValueError("source_refs must contain at least one reference")
        if any(ref.project_id != project for ref in self.source_refs):
            raise ValueError("source_refs project_id must match input project_id")
        object.__setattr__(self, "source_refs", list(self.source_refs))
        digest = _bounded(self.content_hash, "content_hash", 64, required=True).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("content_hash must be a 64-character SHA-256 hex digest")
        object.__setattr__(self, "content_hash", digest)
        if self.problem_signal is not None:
            object.__setattr__(self, "problem_signal", _bounded(self.problem_signal, "problem_signal", MAX_SIGNAL_LENGTH))
        for name in ("mechanism_signals", "implementation_signals", "impact_signals"):
            object.__setattr__(self, name, _strings(getattr(self, name), name, MAX_SIGNAL_LENGTH))
        object.__setattr__(self, "technical_tags", _strings(self.technical_tags, "technical_tags", MAX_TAG_LENGTH, sort=True))
        generated = build_project_evidence_stable_id("pei_", project, {
            "input_type": self.input_type, "title": self.title, "summary": self.summary,
            "problem_signal": self.problem_signal, "mechanism_signals": self.mechanism_signals,
            "implementation_signals": self.implementation_signals, "impact_signals": self.impact_signals,
            "technical_tags": self.technical_tags, "source_refs": self.source_refs, "content_hash": digest,
        })
        object.__setattr__(self, "input_id", self.input_id or generated)


@dataclass(frozen=True)
class ProjectEvidenceFact(ProjectEvidenceModel):
    project_id: str
    mechanism: str
    source_refs: list[EvidenceSourceRef]
    implementation: list[str] = field(default_factory=list)
    problem: str = ""
    safe_impact: list[str] = field(default_factory=list)
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    confidence: Confidence = Confidence.LOW
    metric_support: MetricSupport = MetricSupport.NONE
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    technical_tags: list[str] = field(default_factory=list)
    status: EvidenceStatus = EvidenceStatus.ACCEPTED
    quality_score: float | None = None
    quality_breakdown: dict[str, Any] = field(default_factory=dict)
    evidence_fact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_type", _enum_value(self.evidence_type, EvidenceType, "evidence_type"))
        object.__setattr__(self, "confidence", _enum_value(self.confidence, Confidence, "confidence"))
        object.__setattr__(self, "metric_support", _enum_value(self.metric_support, MetricSupport, "metric_support"))
        object.__setattr__(self, "status", _enum_value(self.status, EvidenceStatus, "status"))
        project = _bounded(self.project_id, "project_id", 300, required=True)
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "mechanism", _bounded(self.mechanism, "mechanism", MAX_DETAIL_LENGTH, required=True))
        object.__setattr__(self, "problem", _bounded(self.problem, "problem", MAX_DETAIL_LENGTH))
        if not self.source_refs:
            raise ValueError("source_refs must contain at least one reference")
        if any(ref.project_id != project for ref in self.source_refs):
            raise ValueError("source_refs project_id must match evidence fact project_id")
        object.__setattr__(self, "source_refs", list(self.source_refs))
        object.__setattr__(self, "implementation", _strings(self.implementation, "implementation", MAX_DETAIL_LENGTH))
        if self.status in (EvidenceStatus.ACCEPTED, EvidenceStatus.SUPPORTING) and not self.implementation:
            raise ValueError("accepted or supporting facts require an implementation detail")
        object.__setattr__(self, "safe_impact", _strings(self.safe_impact, "safe_impact", MAX_CLAIM_LENGTH))
        object.__setattr__(self, "allowed_claims", _strings(self.allowed_claims, "allowed_claims", MAX_CLAIM_LENGTH, sort=True))
        object.__setattr__(self, "forbidden_claims", _strings(self.forbidden_claims, "forbidden_claims", MAX_CLAIM_LENGTH, sort=True))
        object.__setattr__(self, "technical_tags", _strings(self.technical_tags, "technical_tags", MAX_TAG_LENGTH, sort=True))
        if self.quality_score is not None and (isinstance(self.quality_score, bool) or not 0 <= self.quality_score <= 100):
            raise ValueError("quality_score must be between 0 and 100")
        object.__setattr__(self, "quality_breakdown", _json_safe(self.quality_breakdown, path="quality_breakdown"))
        generated = build_project_evidence_stable_id("pef_", project, {
            "problem": self.problem, "mechanism": self.mechanism, "implementation": self.implementation,
            "safe_impact": self.safe_impact, "evidence_type": self.evidence_type,
            "source_refs": self.source_refs, "allowed_claims": self.allowed_claims,
            "forbidden_claims": self.forbidden_claims, "technical_tags": self.technical_tags,
        })
        object.__setattr__(self, "evidence_fact_id", self.evidence_fact_id or generated)


@dataclass(frozen=True)
class ProjectCapabilityFact(ProjectEvidenceModel):
    project_id: str
    capability_type: str
    present: bool
    source_evidence_fact_ids: list[str]
    confidence: Confidence = Confidence.LOW
    mechanisms: list[str] = field(default_factory=list)
    allowed_resume_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    metric_support: MetricSupport = MetricSupport.NONE
    technical_tags: list[str] = field(default_factory=list)
    capability_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _enum_value(self.confidence, Confidence, "confidence"))
        object.__setattr__(self, "metric_support", _enum_value(self.metric_support, MetricSupport, "metric_support"))
        project = _bounded(self.project_id, "project_id", 300, required=True)
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "capability_type", _identifier(self.capability_type, "capability_type"))
        ids = _strings(self.source_evidence_fact_ids, "source_evidence_fact_ids", 100, sort=True)
        if self.present and not ids:
            raise ValueError("present capability requires at least one source evidence fact ID")
        object.__setattr__(self, "source_evidence_fact_ids", ids)
        object.__setattr__(self, "mechanisms", _strings(self.mechanisms, "mechanisms", MAX_DETAIL_LENGTH))
        object.__setattr__(self, "allowed_resume_claims", _strings(self.allowed_resume_claims, "allowed_resume_claims", MAX_CLAIM_LENGTH, sort=True))
        object.__setattr__(self, "forbidden_claims", _strings(self.forbidden_claims, "forbidden_claims", MAX_CLAIM_LENGTH, sort=True))
        object.__setattr__(self, "technical_tags", _strings(self.technical_tags, "technical_tags", MAX_TAG_LENGTH, sort=True))
        generated = build_project_evidence_stable_id("pcf_", project, {"capability_type": self.capability_type, "present": self.present, "source_evidence_fact_ids": ids})
        object.__setattr__(self, "capability_id", self.capability_id or generated)


@dataclass(frozen=True)
class ProjectClaimBoundary(ProjectEvidenceModel):
    project_id: str
    subject_type: ClaimSubjectType
    subject_id: str
    allowed_claims: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    metric_support: MetricSupport = MetricSupport.NONE
    notes: list[str] = field(default_factory=list)
    boundary_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_type", _enum_value(self.subject_type, ClaimSubjectType, "subject_type"))
        object.__setattr__(self, "metric_support", _enum_value(self.metric_support, MetricSupport, "metric_support"))
        project = _bounded(self.project_id, "project_id", 300, required=True)
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "subject_id", _bounded(self.subject_id, "subject_id", 100, required=True))
        allowed = _strings(self.allowed_claims, "allowed_claims", MAX_CLAIM_LENGTH, sort=True)
        forbidden = _strings(self.forbidden_claims, "forbidden_claims", MAX_CLAIM_LENGTH, sort=True)
        conflicts = {item.casefold() for item in allowed} & {item.casefold() for item in forbidden}
        if conflicts:
            raise ValueError("the same normalized claim cannot be both allowed and forbidden")
        object.__setattr__(self, "allowed_claims", allowed)
        object.__setattr__(self, "forbidden_claims", forbidden)
        object.__setattr__(self, "notes", _strings(self.notes, "notes", MAX_NOTES_LENGTH))
        generated = build_project_evidence_stable_id("pcb_", project, {"subject_type": self.subject_type, "subject_id": self.subject_id, "allowed_claims": allowed, "forbidden_claims": forbidden})
        object.__setattr__(self, "boundary_id", self.boundary_id or generated)


@dataclass(frozen=True)
class ProjectEvidencePipelineWarning(ProjectEvidenceModel):
    code: str
    message: str
    project_id: str | None = None
    source_id: str | None = None
    severity: WarningSeverity = WarningSeverity.WARNING

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", _enum_value(self.severity, WarningSeverity, "severity"))
        object.__setattr__(self, "code", _identifier(self.code, "code"))
        object.__setattr__(self, "message", _bounded(self.message, "message", MAX_WARNING_MESSAGE_LENGTH, required=True))
        for name in ("project_id", "source_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _bounded(value, name, 300, required=True))


@dataclass(frozen=True)
class ProjectEvidenceMemory(ProjectEvidenceModel):
    project_id: str
    project_name: str
    source_hashes: dict[str, Any] = field(default_factory=dict)
    evidence_facts: list[ProjectEvidenceFact] = field(default_factory=list)
    capability_facts: list[ProjectCapabilityFact] = field(default_factory=list)
    claim_boundaries: list[ProjectClaimBoundary] = field(default_factory=list)
    quality_summary: dict[str, Any] = field(default_factory=lambda: {"accepted_count": 0, "supporting_count": 0, "weak_count": 0, "rejected_count": 0})
    warnings: list[ProjectEvidencePipelineWarning] = field(default_factory=list)
    schema_version: str = PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION
    project_memory_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION!r}")
        project = _bounded(self.project_id, "project_id", 300, required=True)
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "project_name", _bounded(self.project_name, "project_name", 300, required=True))
        object.__setattr__(self, "source_hashes", _json_safe(self.source_hashes, path="source_hashes"))
        collections = (
            ("evidence_facts", self.evidence_facts, "evidence_fact_id", lambda item: item.evidence_fact_id),
            ("capability_facts", self.capability_facts, "capability_id", lambda item: (item.capability_type, item.capability_id)),
            ("claim_boundaries", self.claim_boundaries, "boundary_id", lambda item: (item.subject_type.value, item.subject_id, item.boundary_id)),
        )
        for name, values, id_name, sort_key in collections:
            if any(item.project_id != project for item in values):
                raise ValueError(f"all {name} project IDs must match parent project_id")
            ids = [getattr(item, id_name) for item in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {id_name} in {name}")
            object.__setattr__(self, name, sorted(list(values), key=sort_key))
        if any(item.project_id not in (None, project) for item in self.warnings):
            raise ValueError("warning project IDs must match parent project_id")
        object.__setattr__(self, "warnings", sorted(list(self.warnings), key=lambda item: (
            item.code, item.project_id or "", item.source_id or "", item.severity.value, item.message,
        )))
        summary = _json_safe(self.quality_summary, path="quality_summary")
        required = {"accepted_count", "supporting_count", "weak_count", "rejected_count"}
        if set(summary) != required or any(isinstance(summary[key], bool) or not isinstance(summary[key], int) or summary[key] < 0 for key in required):
            raise ValueError("quality_summary must contain exactly four non-negative integer counts")
        object.__setattr__(self, "quality_summary", summary)
        generated = build_project_evidence_stable_id("pem_", project, {
            "schema_version": self.schema_version,
            "project_id": project,
            "project_name": self.project_name,
            "source_hashes": self.source_hashes,
            "evidence_facts": self.evidence_facts,
            "capability_facts": self.capability_facts,
            "claim_boundaries": self.claim_boundaries,
            "quality_summary": self.quality_summary,
        })
        object.__setattr__(self, "project_memory_id", self.project_memory_id or generated)


@dataclass(frozen=True)
class ProjectEvidenceBuildResult(ProjectEvidenceModel):
    status: PipelineStatus
    enabled: bool
    projects_processed: int = 0
    evidence_facts_created: int = 0
    capability_facts_created: int = 0
    claim_boundaries_created: int = 0
    items_skipped: int = 0
    memory_written: bool = False
    project_ids: list[str] = field(default_factory=list)
    warnings: list[ProjectEvidencePipelineWarning] = field(default_factory=list)
    errors: list[ProjectEvidencePipelineWarning] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum_value(self.status, PipelineStatus, "status"))
        for name in ("projects_processed", "evidence_facts_created", "capability_facts_created", "claim_boundaries_created", "items_skipped"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "project_ids", _strings(self.project_ids, "project_ids", 300, sort=True))
        object.__setattr__(self, "warnings", list(self.warnings))
        object.__setattr__(self, "errors", list(self.errors))
