"""Strict models and deterministic persistence for Project Capability Memory.

The persistence layer accepts already-built authoritative capability facts.  It
does not run extraction, grouping, scoring, claim-boundary inheritance, fact
building, retrieval, resume generation, or external services.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from backend.project_capability_taxonomy import get_capability_rule
from backend.project_evidence_memory import DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH
from backend.project_evidence_models import (
    PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION,
    Confidence,
    MetricSupport,
    ProjectCapabilityFact,
)


CONFIDENCE_LEVELS = frozenset(item.value for item in Confidence)
METRIC_SUPPORT_LEVELS = frozenset(item.value for item in MetricSupport)

ROOT_DIR = Path(__file__).parents[1]
PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION = "project_capability_memory.v1"
PROJECT_CAPABILITY_MEMORY_PATH = ROOT_DIR / "information" / "project_capability_memory.json"

MAX_IDENTIFIER_LENGTH = 300
MAX_NAME_LENGTH = 300
MAX_SUMMARY_LENGTH = 1_000
MAX_DETAIL_LENGTH = 1_000
MAX_CLAIM_LENGTH = 1_000
MAX_LIST_ITEMS = 200
MAX_METADATA_ITEMS = 100
MAX_METADATA_DEPTH = 6
MAX_METADATA_STRING_LENGTH = 1_000
MAX_CAPABILITY_MEMORY_PROJECTS = 1_000
MAX_CAPABILITY_MEMORY_FACTS = 10_000
MAX_CAPABILITY_MEMORY_WARNINGS = 100
MAX_CAPABILITY_MEMORY_SERIALIZED_SIZE = 10 * 1024 * 1024
MAX_CAPABILITY_MEMORY_VALIDATION_ERRORS = 100

FORBIDDEN_CAPABILITY_METADATA_KEYS = frozenset({
    "rawtext", "rawcontent", "rawdiff", "rawpatch", "patch", "fulldiff",
    "completediff", "sourcecode", "fullsource", "githubcontext", "repositorydump",
    "authorization", "authorizationheader", "accesstoken", "apikey", "token",
    "secret", "credential", "credentials", "password", "privatekey",
    "chainofthought", "reasoning",
})

_TYPE_INPUT_RE = re.compile(r"^[A-Za-z0-9]+(?:[ _-][A-Za-z0-9]+)*$")
_TYPE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_NUMBERED_STAGE_RE = re.compile(r"(?i)phase[ _-]?\d+")
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(?:-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|^\s*bearer\s+[a-z0-9._~+/=-]{12,}\s*$|^(?:gh[oprsu]_|sk-)[a-z0-9_-]{16,}$)"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|private|tmp|var)/)", re.IGNORECASE
)
_RAW_VALUE_RE = re.compile(
    r"(?i)\b(?:raw[_ -]?(?:text|content|patch|diff|github context)|"
    r"authorization[_ -]?header|chain[_ -]?of[_ -]?thought)\s*[:=]"
)
_FORBIDDEN_ARTIFACT_FIELDS = FORBIDDEN_CAPABILITY_METADATA_KEYS | frozenset({
    "generatedat", "writtenat", "updatedat", "buildtimestamp", "runtimeduration",
    "hostname", "workingdirectory", "branchname", "committime", "temporarypath",
    "jobdescription", "resumetemplate",
})
_T = TypeVar("_T", bound="CapabilityModel")


def _text(value: Any, name: str, maximum: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())
    if required and not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    return normalized


def _capability_type(value: Any) -> str:
    candidate = _text(value, "capability_type", MAX_IDENTIFIER_LENGTH, required=True)
    if not _TYPE_INPUT_RE.fullmatch(candidate):
        raise ValueError("capability_type contains unsafe punctuation or display prose")
    normalized = re.sub(r"[ -]+", "_", candidate.casefold())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not _TYPE_RE.fullmatch(normalized) or _NUMBERED_STAGE_RE.search(normalized):
        raise ValueError("capability_type must be a semantic lowercase snake_case identifier")
    return normalized


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0")
    return normalized


def _strings(
    values: Any,
    name: str,
    maximum: int,
    *,
    required: bool = False,
    ordered: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if len(values) > MAX_LIST_ITEMS:
        raise ValueError(f"{name} exceeds maximum item count {MAX_LIST_ITEMS}")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _text(value, name, maximum, required=True)
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    if required and not result:
        raise ValueError(f"{name} must contain at least one value")
    if not ordered:
        result.sort(key=lambda item: (item.casefold(), item))
    return tuple(result)


def _metadata_key(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} keys must be strings")
    key = value.strip()
    if not key:
        raise ValueError(f"{path} keys must not be blank")
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    if normalized in FORBIDDEN_CAPABILITY_METADATA_KEYS:
        raise ValueError(f"forbidden capability metadata field: {key}")
    return key


def _freeze_json(value: Any, *, path: str = "metadata", depth: int = 0) -> Any:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"{path} exceeds maximum nesting depth {MAX_METADATA_DEPTH}")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, str):
        normalized = _text(value, path, MAX_METADATA_STRING_LENGTH)
        if _SENSITIVE_VALUE_RE.search(normalized):
            raise ValueError(f"{path} contains a sensitive value")
        return normalized
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_ITEMS:
            raise ValueError(f"{path} exceeds maximum item count {MAX_METADATA_ITEMS}")
        items: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _metadata_key(raw_key, path)
            items[key] = _freeze_json(item, path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType({key: items[key] for key in sorted(items)})
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST_ITEMS:
            raise ValueError(f"{path} exceeds maximum item count {MAX_LIST_ITEMS}")
        return tuple(_freeze_json(item, path=f"{path}[]", depth=depth + 1) for item in value)
    raise TypeError(f"{path} contains non-JSON-safe value {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class CapabilityModel:
    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for model_field in fields(self):
            value = getattr(self, model_field.name)
            output[model_field.name] = list(value) if isinstance(value, tuple) else _thaw_json(value)
        return output

    def to_safe_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls: type[_T], payload: Mapping[str, Any]) -> _T:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects an object")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CapabilityCandidate(CapabilityModel):
    project_id: str
    capability_type: str
    supporting_evidence_ids: tuple[str, ...]
    supporting_signals: tuple[str, ...]
    conflicting_signals: tuple[str, ...]
    candidate_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", MAX_IDENTIFIER_LENGTH, required=True))
        object.__setattr__(self, "capability_type", _capability_type(self.capability_type))
        object.__setattr__(self, "supporting_evidence_ids", _strings(self.supporting_evidence_ids, "supporting_evidence_ids", MAX_IDENTIFIER_LENGTH, required=True))
        object.__setattr__(self, "supporting_signals", _strings(self.supporting_signals, "supporting_signals", MAX_DETAIL_LENGTH))
        object.__setattr__(self, "conflicting_signals", _strings(self.conflicting_signals, "conflicting_signals", MAX_DETAIL_LENGTH))
        object.__setattr__(self, "candidate_score", _score(self.candidate_score, "candidate_score"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


def normalize_capability_candidate(value: CapabilityCandidate | Mapping[str, Any]) -> CapabilityCandidate:
    return CapabilityCandidate.from_dict(value) if isinstance(value, Mapping) else CapabilityCandidate.from_dict(value.to_dict())


def validate_capability_candidate(value: CapabilityCandidate) -> CapabilityCandidate:
    if not isinstance(value, CapabilityCandidate):
        raise TypeError("value must be a CapabilityCandidate")
    return normalize_capability_candidate(value)


def normalize_project_capability_fact(
    value: ProjectCapabilityFact | Mapping[str, Any],
) -> ProjectCapabilityFact:
    """Normalize through the existing authoritative evidence-memory fact model."""
    if isinstance(value, ProjectCapabilityFact):
        return ProjectCapabilityFact.from_dict(value.to_dict())
    return ProjectCapabilityFact.from_dict(value)


def validate_project_capability_fact(value: ProjectCapabilityFact) -> ProjectCapabilityFact:
    if not isinstance(value, ProjectCapabilityFact):
        raise TypeError("value must be a ProjectCapabilityFact")
    return normalize_project_capability_fact(value)


class ProjectCapabilityMemoryIntegrityError(ValueError):
    """Raised when a capability-memory schema or identity invariant fails."""


def _nonnegative_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.casefold()):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value.casefold()


def _strict_fields(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be an object")
    unknown = set(payload) - allowed
    missing = allowed - set(payload)
    if unknown or missing:
        raise ValueError(f"invalid {name} fields")


def _summary_strings(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if len(values) > MAX_LIST_ITEMS:
        raise ValueError(f"{name} exceeds maximum item count {MAX_LIST_ITEMS}")
    normalized = tuple(
        _text(value, name, MAX_IDENTIFIER_LENGTH, required=True) for value in values
    )
    if len({value.casefold() for value in normalized}) != len(normalized):
        raise ValueError(f"{name} contains duplicate values")
    return normalized


def _safe_count_mapping(values: Any, name: str) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be an object")
    if len(values) > MAX_METADATA_ITEMS:
        raise ValueError(f"{name} exceeds maximum item count {MAX_METADATA_ITEMS}")
    output: dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _SAFE_CODE_RE.fullmatch(key):
            raise ValueError(f"{name} keys must be safe identifiers")
        output[key] = _nonnegative_count(value, name)
    return MappingProxyType({key: output[key] for key in sorted(output)})


@dataclass(frozen=True)
class ProjectCapabilitySourceArtifact:
    schema_version: str
    content_hash: str
    file_sha256: str | None
    project_count: int
    evidence_fact_count: int
    claim_boundary_count: int

    def __post_init__(self) -> None:
        schema = _text(self.schema_version, "source schema_version", MAX_IDENTIFIER_LENGTH, required=True)
        if schema != PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported source artifact schema version")
        object.__setattr__(self, "schema_version", schema)
        object.__setattr__(self, "content_hash", _sha256(self.content_hash, "source content_hash"))
        object.__setattr__(self, "file_sha256", _sha256(self.file_sha256, "source file_sha256", optional=True))
        for name in ("project_count", "evidence_fact_count", "claim_boundary_count"):
            object.__setattr__(self, name, _nonnegative_count(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "file_sha256": self.file_sha256,
            "project_count": self.project_count,
            "evidence_fact_count": self.evidence_fact_count,
            "claim_boundary_count": self.claim_boundary_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectCapabilitySourceArtifact":
        allowed = {
            "schema_version", "content_hash", "file_sha256", "project_count",
            "evidence_fact_count", "claim_boundary_count",
        }
        _strict_fields(payload, allowed, "source_artifact")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ProjectCapabilityProjectSummary:
    project_id: str
    capability_fact_ids: tuple[str, ...]
    capability_types: tuple[str, ...]
    capability_fact_count: int
    confirmed_capability_count: int
    metric_supported_capability_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _text(
            self.project_id, "project_id", MAX_IDENTIFIER_LENGTH, required=True
        ))
        object.__setattr__(self, "capability_fact_ids", _summary_strings(
            self.capability_fact_ids, "capability_fact_ids"
        ))
        object.__setattr__(self, "capability_types", _summary_strings(
            self.capability_types, "capability_types"
        ))
        for name in (
            "capability_fact_count", "confirmed_capability_count",
            "metric_supported_capability_count",
        ):
            object.__setattr__(self, name, _nonnegative_count(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "capability_fact_ids": list(self.capability_fact_ids),
            "capability_types": list(self.capability_types),
            "capability_fact_count": self.capability_fact_count,
            "confirmed_capability_count": self.confirmed_capability_count,
            "metric_supported_capability_count": self.metric_supported_capability_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectCapabilityProjectSummary":
        allowed = {
            "project_id", "capability_fact_ids", "capability_types",
            "capability_fact_count", "confirmed_capability_count",
            "metric_supported_capability_count",
        }
        _strict_fields(payload, allowed, "project_summary")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ProjectCapabilityMemoryDiagnostics:
    source_project_count: int
    source_evidence_fact_count: int
    source_claim_boundary_count: int
    project_count: int
    capability_fact_count: int
    projects_with_capabilities: int
    projects_without_capabilities: int
    capability_type_counts: Mapping[str, int]
    confidence_counts: Mapping[str, int]
    metric_support_counts: Mapping[str, int]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_project_count", "source_evidence_fact_count",
            "source_claim_boundary_count", "project_count", "capability_fact_count",
            "projects_with_capabilities", "projects_without_capabilities",
        ):
            object.__setattr__(self, name, _nonnegative_count(getattr(self, name), name))
        for name in ("capability_type_counts", "confidence_counts", "metric_support_counts"):
            object.__setattr__(self, name, _safe_count_mapping(getattr(self, name), name))
        warnings = _summary_strings(self.warnings, "warnings")
        if any(not _SAFE_CODE_RE.fullmatch(value) for value in warnings):
            raise ValueError("warnings must contain safe codes only")
        object.__setattr__(self, "warnings", tuple(sorted(warnings)[:MAX_CAPABILITY_MEMORY_WARNINGS]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_project_count": self.source_project_count,
            "source_evidence_fact_count": self.source_evidence_fact_count,
            "source_claim_boundary_count": self.source_claim_boundary_count,
            "project_count": self.project_count,
            "capability_fact_count": self.capability_fact_count,
            "projects_with_capabilities": self.projects_with_capabilities,
            "projects_without_capabilities": self.projects_without_capabilities,
            "capability_type_counts": dict(self.capability_type_counts),
            "confidence_counts": dict(self.confidence_counts),
            "metric_support_counts": dict(self.metric_support_counts),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectCapabilityMemoryDiagnostics":
        allowed = {
            "source_project_count", "source_evidence_fact_count", "source_claim_boundary_count",
            "project_count", "capability_fact_count", "projects_with_capabilities",
            "projects_without_capabilities", "capability_type_counts", "confidence_counts",
            "metric_support_counts", "warnings",
        }
        _strict_fields(payload, allowed, "diagnostics")
        values = dict(payload)
        values["warnings"] = tuple(values["warnings"])
        return cls(**values)


@dataclass(frozen=True)
class ProjectCapabilityMemoryValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectCapabilityMemory:
    schema_version: str
    source_artifact: ProjectCapabilitySourceArtifact
    projects: tuple[ProjectCapabilityProjectSummary, ...]
    capability_facts: tuple[ProjectCapabilityFact, ...]
    diagnostics: ProjectCapabilityMemoryDiagnostics
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _text(
            self.schema_version, "schema_version", MAX_IDENTIFIER_LENGTH, required=True
        ))
        if not isinstance(self.source_artifact, ProjectCapabilitySourceArtifact):
            raise TypeError("source_artifact must be a ProjectCapabilitySourceArtifact")
        if not isinstance(self.projects, (list, tuple)):
            raise TypeError("projects must be a list or tuple")
        if not isinstance(self.capability_facts, (list, tuple)):
            raise TypeError("capability_facts must be a list or tuple")
        if any(not isinstance(item, ProjectCapabilityProjectSummary) for item in self.projects):
            raise TypeError("projects must contain ProjectCapabilityProjectSummary values")
        if any(not isinstance(item, ProjectCapabilityFact) for item in self.capability_facts):
            raise TypeError("capability_facts must contain ProjectCapabilityFact values")
        if not isinstance(self.diagnostics, ProjectCapabilityMemoryDiagnostics):
            raise TypeError("diagnostics must be a ProjectCapabilityMemoryDiagnostics")
        object.__setattr__(self, "source_artifact", ProjectCapabilitySourceArtifact.from_dict(
            self.source_artifact.to_dict()
        ))
        object.__setattr__(self, "projects", tuple(
            ProjectCapabilityProjectSummary.from_dict(item.to_dict()) for item in self.projects
        ))
        object.__setattr__(self, "capability_facts", tuple(
            ProjectCapabilityFact.from_dict(item.to_dict()) for item in self.capability_facts
        ))
        object.__setattr__(self, "diagnostics", ProjectCapabilityMemoryDiagnostics.from_dict(
            self.diagnostics.to_dict()
        ))
        object.__setattr__(self, "content_hash", _text(
            self.content_hash, "content_hash", 64
        ).casefold())

    def to_dict(self) -> dict[str, Any]:
        return _memory_payload(self, include_hash=True)

    def to_safe_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectCapabilityMemory":
        allowed = {
            "schema_version", "source_artifact", "projects", "capability_facts",
            "diagnostics", "content_hash",
        }
        _strict_fields(payload, allowed, "project_capability_memory")
        if payload.get("schema_version") != PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION:
            raise ProjectCapabilityMemoryIntegrityError("unsupported_schema_version")
        try:
            memory = cls(
                schema_version=payload["schema_version"],
                source_artifact=ProjectCapabilitySourceArtifact.from_dict(payload["source_artifact"]),
                projects=tuple(ProjectCapabilityProjectSummary.from_dict(item) for item in payload["projects"]),
                capability_facts=tuple(ProjectCapabilityFact.from_dict(item) for item in payload["capability_facts"]),
                diagnostics=ProjectCapabilityMemoryDiagnostics.from_dict(payload["diagnostics"]),
                content_hash=payload["content_hash"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectCapabilityMemoryIntegrityError("invalid_memory_payload") from exc
        validation = validate_project_capability_memory(memory)
        if not validation.valid:
            raise ProjectCapabilityMemoryIntegrityError(validation.errors[0])
        return memory


@dataclass(frozen=True)
class ProjectCapabilityMemoryLoadResult:
    status: str
    memory: ProjectCapabilityMemory | None
    validation: ProjectCapabilityMemoryValidationReport


@dataclass(frozen=True)
class ProjectCapabilityMemoryPersistenceReport:
    status: str
    path: str
    schema_version: str
    content_hash: str
    project_count: int
    capability_fact_count: int
    bytes_written: int
    previous_artifact_preserved: bool
    round_trip_validated: bool
    warnings: tuple[str, ...] = ()


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True, "allow_nan": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def _reject_unsafe_artifact_content(value: Any, *, field_name: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProjectCapabilityMemoryIntegrityError("non_string_artifact_field")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if normalized in _FORBIDDEN_ARTIFACT_FIELDS:
                raise ProjectCapabilityMemoryIntegrityError("forbidden_artifact_field")
            _reject_unsafe_artifact_content(item, field_name=key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_unsafe_artifact_content(item, field_name=field_name)
        return
    if isinstance(value, str) and (
        _SENSITIVE_VALUE_RE.search(value)
        or _ABSOLUTE_LOCAL_PATH_RE.search(value)
        or _RAW_VALUE_RE.search(value)
    ):
        raise ProjectCapabilityMemoryIntegrityError(f"unsafe_artifact_value:{field_name}")


def _canonical_fact(fact: ProjectCapabilityFact) -> ProjectCapabilityFact:
    if not isinstance(fact, ProjectCapabilityFact):
        raise TypeError("capability_facts must contain ProjectCapabilityFact values")
    clone = ProjectCapabilityFact.from_dict(fact.to_dict())
    definition = get_capability_rule(clone.capability_type)
    if definition is None or definition.capability_type != clone.capability_type:
        raise ProjectCapabilityMemoryIntegrityError("noncanonical_capability_type")
    if clone.present is not True:
        raise ProjectCapabilityMemoryIntegrityError("capability_fact_must_be_present")
    authoritative_id = ProjectCapabilityFact(
        project_id=clone.project_id,
        capability_type=clone.capability_type,
        present=clone.present,
        source_evidence_fact_ids=list(clone.source_evidence_fact_ids),
    ).capability_id
    if clone.capability_id != authoritative_id:
        raise ProjectCapabilityMemoryIntegrityError("invalid_capability_id")
    normalized = ProjectCapabilityFact(
        project_id=clone.project_id,
        capability_type=clone.capability_type,
        present=True,
        source_evidence_fact_ids=sorted(clone.source_evidence_fact_ids),
        confidence=clone.confidence,
        mechanisms=sorted(clone.mechanisms, key=lambda item: (item.casefold(), item)),
        allowed_resume_claims=sorted(clone.allowed_resume_claims, key=lambda item: (item.casefold(), item)),
        forbidden_claims=sorted(clone.forbidden_claims, key=lambda item: (item.casefold(), item)),
        metric_support=clone.metric_support,
        technical_tags=sorted(clone.technical_tags, key=lambda item: (item.casefold(), item)),
    )
    _reject_unsafe_artifact_content(normalized.to_dict())
    return normalized


def _derive_project_summaries(
    project_ids: Sequence[str], facts: Sequence[ProjectCapabilityFact]
) -> tuple[ProjectCapabilityProjectSummary, ...]:
    by_project: dict[str, list[ProjectCapabilityFact]] = {project_id: [] for project_id in project_ids}
    for fact in facts:
        if fact.project_id not in by_project:
            raise ProjectCapabilityMemoryIntegrityError("capability_fact_project_not_in_source")
        by_project[fact.project_id].append(fact)
    summaries = []
    for project_id in sorted(by_project, key=lambda item: (item.casefold(), item)):
        project_facts = sorted(
            by_project[project_id], key=lambda item: (item.capability_type, item.capability_id)
        )
        summaries.append(ProjectCapabilityProjectSummary(
            project_id=project_id,
            capability_fact_ids=tuple(fact.capability_id for fact in project_facts),
            capability_types=tuple(fact.capability_type for fact in project_facts),
            capability_fact_count=len(project_facts),
            confirmed_capability_count=sum(fact.present for fact in project_facts),
            metric_supported_capability_count=sum(
                fact.metric_support is not MetricSupport.NONE for fact in project_facts
            ),
        ))
    return tuple(summaries)


def _derive_diagnostics(
    source: ProjectCapabilitySourceArtifact,
    projects: Sequence[ProjectCapabilityProjectSummary],
    facts: Sequence[ProjectCapabilityFact],
) -> ProjectCapabilityMemoryDiagnostics:
    capability_type_counts: dict[str, int] = {}
    confidence_counts = {item.value: 0 for item in Confidence}
    metric_support_counts = {item.value: 0 for item in MetricSupport}
    for fact in facts:
        capability_type_counts[fact.capability_type] = capability_type_counts.get(fact.capability_type, 0) + 1
        confidence_counts[fact.confidence.value] += 1
        metric_support_counts[fact.metric_support.value] += 1
    projects_with = sum(bool(project.capability_fact_count) for project in projects)
    warnings: set[str] = set()
    if not facts:
        warnings.add("capability_facts_empty")
    if projects_with != len(projects):
        warnings.add("projects_without_verified_capabilities")
    if source.file_sha256 is None:
        warnings.add("source_lineage_missing_file_sha256")
    return ProjectCapabilityMemoryDiagnostics(
        source_project_count=source.project_count,
        source_evidence_fact_count=source.evidence_fact_count,
        source_claim_boundary_count=source.claim_boundary_count,
        project_count=len(projects),
        capability_fact_count=len(facts),
        projects_with_capabilities=projects_with,
        projects_without_capabilities=len(projects) - projects_with,
        capability_type_counts=capability_type_counts,
        confidence_counts=confidence_counts,
        metric_support_counts=metric_support_counts,
        warnings=tuple(warnings),
    )


def _memory_payload(memory: ProjectCapabilityMemory, *, include_hash: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": memory.schema_version,
        "source_artifact": memory.source_artifact.to_dict(),
        "projects": [project.to_dict() for project in memory.projects],
        "capability_facts": [fact.to_dict() for fact in memory.capability_facts],
        "diagnostics": memory.diagnostics.to_dict(),
    }
    if include_hash:
        payload["content_hash"] = memory.content_hash
    return payload


def compute_project_capability_memory_content_hash(memory: ProjectCapabilityMemory) -> str:
    if not isinstance(memory, ProjectCapabilityMemory):
        raise TypeError("memory must be a ProjectCapabilityMemory")
    return hashlib.sha256(_canonical_json_bytes(_memory_payload(memory, include_hash=False))).hexdigest()


def build_project_capability_memory(
    *,
    source_artifact: ProjectCapabilitySourceArtifact,
    source_project_ids: Sequence[str],
    capability_facts: Sequence[ProjectCapabilityFact],
    diagnostics: ProjectCapabilityMemoryDiagnostics | Mapping[str, Any] | None = None,
) -> ProjectCapabilityMemory:
    """Build and hash capability memory from already-constructed Facts only."""

    if not isinstance(source_artifact, ProjectCapabilitySourceArtifact):
        raise TypeError("source_artifact must be a ProjectCapabilitySourceArtifact")
    if not isinstance(source_project_ids, Sequence) or isinstance(source_project_ids, (str, bytes)):
        raise TypeError("source_project_ids must be a sequence")
    raw_project_ids = tuple(
        _text(value, "source_project_ids", MAX_IDENTIFIER_LENGTH, required=True)
        for value in source_project_ids
    )
    if len(raw_project_ids) > MAX_CAPABILITY_MEMORY_PROJECTS:
        raise ProjectCapabilityMemoryIntegrityError("maximum_project_count_exceeded")
    if len({value.casefold() for value in raw_project_ids}) != len(raw_project_ids):
        raise ProjectCapabilityMemoryIntegrityError("duplicate_source_project_id")
    project_ids = tuple(sorted(raw_project_ids, key=lambda item: (item.casefold(), item)))
    if source_artifact.project_count != len(project_ids):
        raise ProjectCapabilityMemoryIntegrityError("source_project_count_mismatch")
    if not isinstance(capability_facts, Sequence) or isinstance(capability_facts, (str, bytes)):
        raise TypeError("capability_facts must be a sequence")
    if len(capability_facts) > MAX_CAPABILITY_MEMORY_FACTS:
        raise ProjectCapabilityMemoryIntegrityError("maximum_capability_fact_count_exceeded")

    by_id: dict[str, tuple[str, ProjectCapabilityFact]] = {}
    by_project_type: dict[tuple[str, str], str] = {}
    for value in capability_facts:
        fact = _canonical_fact(value)
        payload = fact.to_json()
        previous = by_id.get(fact.capability_id)
        if previous is not None and previous[0] != payload:
            raise ProjectCapabilityMemoryIntegrityError("conflicting_capability_id")
        identity = (fact.project_id, fact.capability_type)
        previous_id = by_project_type.get(identity)
        if previous_id is not None and previous_id != fact.capability_id:
            raise ProjectCapabilityMemoryIntegrityError("conflicting_project_capability_identity")
        by_id[fact.capability_id] = (payload, fact)
        by_project_type[identity] = fact.capability_id
    facts = tuple(sorted(
        (item[1] for item in by_id.values()),
        key=lambda item: (item.project_id.casefold(), item.project_id, item.capability_type, item.capability_id),
    ))
    projects = _derive_project_summaries(project_ids, facts)
    derived_diagnostics = _derive_diagnostics(source_artifact, projects, facts)
    if diagnostics is not None:
        supplied = (
            diagnostics
            if isinstance(diagnostics, ProjectCapabilityMemoryDiagnostics)
            else ProjectCapabilityMemoryDiagnostics.from_dict(diagnostics)
        )
        if supplied != derived_diagnostics:
            raise ProjectCapabilityMemoryIntegrityError("diagnostics_mismatch")
    memory = ProjectCapabilityMemory(
        schema_version=PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION,
        source_artifact=source_artifact,
        projects=projects,
        capability_facts=facts,
        diagnostics=derived_diagnostics,
        content_hash="",
    )
    memory = replace(memory, content_hash=compute_project_capability_memory_content_hash(memory))
    validation = validate_project_capability_memory(memory)
    if not validation.valid:
        raise ProjectCapabilityMemoryIntegrityError(validation.errors[0])
    return memory


def _bounded_validation_errors(errors: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(errors))[:MAX_CAPABILITY_MEMORY_VALIDATION_ERRORS])


def validate_project_capability_memory(
    memory: ProjectCapabilityMemory,
) -> ProjectCapabilityMemoryValidationReport:
    errors: set[str] = set()
    if not isinstance(memory, ProjectCapabilityMemory):
        return ProjectCapabilityMemoryValidationReport(False, ("invalid_memory_type",))
    if memory.schema_version != PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION:
        errors.add("unsupported_schema_version")
    source = memory.source_artifact
    if not isinstance(source, ProjectCapabilitySourceArtifact):
        errors.add("invalid_source_artifact")
    projects = memory.projects
    facts = memory.capability_facts
    if not isinstance(projects, tuple):
        errors.add("projects_must_be_tuple")
        projects = tuple(projects) if isinstance(projects, (list, tuple)) else ()
    if not isinstance(facts, tuple):
        errors.add("capability_facts_must_be_tuple")
        facts = tuple(facts) if isinstance(facts, (list, tuple)) else ()
    if len(projects) > MAX_CAPABILITY_MEMORY_PROJECTS:
        errors.add("maximum_project_count_exceeded")
    if len(facts) > MAX_CAPABILITY_MEMORY_FACTS:
        errors.add("maximum_capability_fact_count_exceeded")
    if any(not isinstance(item, ProjectCapabilityProjectSummary) for item in projects):
        errors.add("invalid_project_summary")
    if any(not isinstance(item, ProjectCapabilityFact) for item in facts):
        errors.add("invalid_capability_fact")
    valid_projects = tuple(item for item in projects if isinstance(item, ProjectCapabilityProjectSummary))
    valid_facts = tuple(item for item in facts if isinstance(item, ProjectCapabilityFact))
    ordered_projects = tuple(sorted(valid_projects, key=lambda item: (item.project_id.casefold(), item.project_id)))
    ordered_facts = tuple(sorted(
        valid_facts,
        key=lambda item: (item.project_id.casefold(), item.project_id, item.capability_type, item.capability_id),
    ))
    if valid_projects != ordered_projects:
        errors.add("non_deterministic_project_order")
    if valid_facts != ordered_facts:
        errors.add("non_deterministic_fact_order")
    project_ids = [project.project_id for project in valid_projects]
    if len(set(project_ids)) != len(project_ids):
        errors.add("duplicate_project_summary")
    fact_ids: dict[str, str] = {}
    project_types: dict[tuple[str, str], str] = {}
    for fact in valid_facts:
        try:
            canonical = _canonical_fact(fact)
            if canonical.to_json() != fact.to_json():
                errors.add("noncanonical_capability_fact")
        except (TypeError, ValueError, ProjectCapabilityMemoryIntegrityError) as exc:
            code = str(exc)
            errors.add(code if _SAFE_CODE_RE.fullmatch(code) else "invalid_capability_fact")
            continue
        payload = canonical.to_json()
        previous = fact_ids.get(canonical.capability_id)
        if previous is not None:
            errors.add("duplicate_capability_id" if previous == payload else "conflicting_capability_id")
        fact_ids[canonical.capability_id] = payload
        identity = (canonical.project_id, canonical.capability_type)
        previous_id = project_types.get(identity)
        if previous_id is not None and previous_id != canonical.capability_id:
            errors.add("conflicting_project_capability_identity")
        project_types[identity] = canonical.capability_id
    try:
        expected_projects = _derive_project_summaries(project_ids, valid_facts)
        if valid_projects != expected_projects:
            errors.add("project_fact_referential_integrity_failure")
    except (TypeError, ValueError, ProjectCapabilityMemoryIntegrityError):
        errors.add("project_fact_referential_integrity_failure")
    if source.project_count != len(valid_projects):
        errors.add("source_project_count_mismatch")
    try:
        expected_diagnostics = _derive_diagnostics(source, valid_projects, valid_facts)
        if memory.diagnostics != expected_diagnostics:
            errors.add("diagnostics_mismatch")
    except (TypeError, ValueError):
        errors.add("invalid_diagnostics")
    try:
        payload = _memory_payload(memory, include_hash=True)
        _reject_unsafe_artifact_content(payload)
        if len(_canonical_json_bytes(payload, pretty=True)) + 1 > MAX_CAPABILITY_MEMORY_SERIALIZED_SIZE:
            errors.add("maximum_serialized_size_exceeded")
    except (TypeError, ValueError, ProjectCapabilityMemoryIntegrityError):
        errors.add("unsafe_artifact_content")
    if not isinstance(memory.content_hash, str) or not _SHA256_RE.fullmatch(memory.content_hash):
        errors.add("invalid_content_hash")
    else:
        try:
            if memory.content_hash != compute_project_capability_memory_content_hash(memory):
                errors.add("content_hash_mismatch")
        except (TypeError, ValueError):
            errors.add("content_hash_unavailable")
    final_errors = _bounded_validation_errors(errors)
    return ProjectCapabilityMemoryValidationReport(not final_errors, final_errors)


def serialize_project_capability_memory(memory: ProjectCapabilityMemory) -> bytes:
    validation = validate_project_capability_memory(memory)
    if not validation.valid:
        raise ProjectCapabilityMemoryIntegrityError(f"invalid_memory:{validation.errors[0]}")
    payload = memory.to_dict()
    _reject_unsafe_artifact_content(payload)
    serialized = _canonical_json_bytes(payload, pretty=True) + b"\n"
    if len(serialized) > MAX_CAPABILITY_MEMORY_SERIALIZED_SIZE:
        raise ProjectCapabilityMemoryIntegrityError("maximum_serialized_size_exceeded")
    return serialized


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise _DuplicateJsonKeyError("duplicate_json_key")
        output[key] = value
    return output


def load_project_capability_memory(
    path: Path | str = PROJECT_CAPABILITY_MEMORY_PATH,
) -> ProjectCapabilityMemoryLoadResult:
    destination = Path(path)
    if not destination.exists():
        return ProjectCapabilityMemoryLoadResult(
            "missing", None, ProjectCapabilityMemoryValidationReport(False, ("artifact_missing",))
        )
    if destination.is_dir():
        return ProjectCapabilityMemoryLoadResult(
            "invalid", None, ProjectCapabilityMemoryValidationReport(False, ("destination_is_directory",))
        )
    try:
        if destination.stat().st_size > MAX_CAPABILITY_MEMORY_SERIALIZED_SIZE:
            raise ProjectCapabilityMemoryIntegrityError("maximum_serialized_size_exceeded")
        payload = json.loads(
            destination.read_bytes().decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except Exception:
        return ProjectCapabilityMemoryLoadResult(
            "invalid", None, ProjectCapabilityMemoryValidationReport(False, ("malformed_artifact",))
        )
    if not isinstance(payload, Mapping):
        return ProjectCapabilityMemoryLoadResult(
            "invalid", None, ProjectCapabilityMemoryValidationReport(False, ("invalid_memory_shape",))
        )
    if payload.get("schema_version") != PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION:
        return ProjectCapabilityMemoryLoadResult(
            "unsupported_version", None,
            ProjectCapabilityMemoryValidationReport(False, ("unsupported_schema_version",)),
        )
    try:
        memory = ProjectCapabilityMemory.from_dict(payload)
    except (ProjectCapabilityMemoryIntegrityError, TypeError, ValueError) as exc:
        code = str(exc)
        status = "hash_mismatch" if code in {"content_hash_mismatch", "invalid_content_hash"} else "invalid"
        return ProjectCapabilityMemoryLoadResult(
            status, None,
            ProjectCapabilityMemoryValidationReport(False, (code if _SAFE_CODE_RE.fullmatch(code) else "invalid_memory_payload",)),
        )
    status = "empty" if not memory.capability_facts else "ready"
    return ProjectCapabilityMemoryLoadResult(
        status, memory, ProjectCapabilityMemoryValidationReport(True, ())
    )


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(
            str(right.resolve(strict=False))
        )
    except OSError:
        return os.path.normcase(str(left.absolute())) == os.path.normcase(str(right.absolute()))


def _write_staged_project_capability_json(path: Path, payload: dict[str, Any]) -> None:
    from backend.project_change_memory import atomic_write_json

    atomic_write_json(path, payload)
    canonical = _canonical_json_bytes(payload, pretty=True) + b"\n"
    if path.read_bytes() != canonical:
        with path.open("wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())


def _replace_staged_project_capability_artifact(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)


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


def _persistence_report(
    memory: ProjectCapabilityMemory,
    destination: Path,
    *,
    status: str,
    bytes_written: int = 0,
    previous_artifact_preserved: bool = False,
    round_trip_validated: bool = False,
    warnings: Iterable[str] = (),
) -> ProjectCapabilityMemoryPersistenceReport:
    return ProjectCapabilityMemoryPersistenceReport(
        status=status,
        path=str(destination),
        schema_version=memory.schema_version,
        content_hash=memory.content_hash,
        project_count=len(memory.projects),
        capability_fact_count=len(memory.capability_facts),
        bytes_written=bytes_written,
        previous_artifact_preserved=previous_artifact_preserved,
        round_trip_validated=round_trip_validated,
        warnings=tuple(sorted(set(warnings))),
    )


def persist_project_capability_memory(
    memory: ProjectCapabilityMemory,
    path: Path | str = PROJECT_CAPABILITY_MEMORY_PATH,
) -> ProjectCapabilityMemoryPersistenceReport:
    """Validate, stage, round-trip, and atomically replace one capability artifact."""

    destination = Path(path)
    if _same_resolved_path(destination, DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH):
        return _persistence_report(
            memory, destination, status="failed", previous_artifact_preserved=destination.exists(),
            warnings=("upstream_artifact_path_forbidden",),
        )
    validation = validate_project_capability_memory(memory)
    if not validation.valid:
        return _persistence_report(
            memory, destination, status="failed", previous_artifact_preserved=destination.exists(),
            warnings=("invalid_memory",),
        )
    serialized = serialize_project_capability_memory(memory)
    existed = destination.exists()
    if existed:
        existing = load_project_capability_memory(destination)
        if existing.status not in {"ready", "empty"} or existing.memory is None:
            return _persistence_report(
                memory, destination, status="failed", previous_artifact_preserved=True,
                warnings=("invalid_existing_artifact",),
            )
        if existing.memory == memory:
            return _persistence_report(
                memory, destination, status="unchanged", previous_artifact_preserved=True,
                round_trip_validated=True,
            )
    if destination.exists() and destination.is_dir():
        return _persistence_report(
            memory, destination, status="failed", previous_artifact_preserved=True,
            warnings=("destination_is_directory",),
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _persistence_report(
            memory, destination, status="failed", previous_artifact_preserved=existed,
            warnings=("parent_directory_create_failed",),
        )
    staged: Path | None = None
    try:
        descriptor, staged_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".stage"
        )
        os.close(descriptor)
        staged = Path(staged_name)
        staged.unlink(missing_ok=True)
        _write_staged_project_capability_json(staged, memory.to_dict())
        staged_load = load_project_capability_memory(staged)
        expected_status = "empty" if not memory.capability_facts else "ready"
        if staged_load.status != expected_status or staged_load.memory != memory:
            return _persistence_report(
                memory, destination, status="failed", previous_artifact_preserved=existed,
                warnings=("temporary_round_trip_failed",),
            )
        _replace_staged_project_capability_artifact(staged, destination)
        staged = None
        _sync_parent_directory(destination.parent)
        return _persistence_report(
            memory, destination, status="updated" if existed else "created",
            bytes_written=len(serialized), round_trip_validated=True,
        )
    except Exception:
        return _persistence_report(
            memory, destination, status="failed", previous_artifact_preserved=existed,
            warnings=("atomic_write_failed",),
        )
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "CONFIDENCE_LEVELS",
    "FORBIDDEN_CAPABILITY_METADATA_KEYS",
    "METRIC_SUPPORT_LEVELS",
    "PROJECT_CAPABILITY_MEMORY_PATH",
    "PROJECT_CAPABILITY_MEMORY_SCHEMA_VERSION",
    "CapabilityCandidate",
    "ProjectCapabilityFact",
    "ProjectCapabilityMemory",
    "ProjectCapabilityMemoryDiagnostics",
    "ProjectCapabilityMemoryIntegrityError",
    "ProjectCapabilityMemoryLoadResult",
    "ProjectCapabilityMemoryPersistenceReport",
    "ProjectCapabilityMemoryValidationReport",
    "ProjectCapabilityProjectSummary",
    "ProjectCapabilitySourceArtifact",
    "build_project_capability_memory",
    "compute_project_capability_memory_content_hash",
    "load_project_capability_memory",
    "normalize_capability_candidate",
    "normalize_project_capability_fact",
    "persist_project_capability_memory",
    "serialize_project_capability_memory",
    "validate_capability_candidate",
    "validate_project_capability_memory",
    "validate_project_capability_fact",
]
