"""Strict model contracts for future Project Capability Memory processing.

This module defines values only.  It performs no extraction, grouping, scoring,
persistence, pipeline orchestration, retrieval, or resume generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeVar

from backend.project_evidence_models import (
    Confidence,
    MetricSupport,
    ProjectCapabilityFact,
)


CONFIDENCE_LEVELS = frozenset(item.value for item in Confidence)
METRIC_SUPPORT_LEVELS = frozenset(item.value for item in MetricSupport)

MAX_IDENTIFIER_LENGTH = 300
MAX_NAME_LENGTH = 300
MAX_SUMMARY_LENGTH = 1_000
MAX_DETAIL_LENGTH = 1_000
MAX_CLAIM_LENGTH = 1_000
MAX_LIST_ITEMS = 200
MAX_METADATA_ITEMS = 100
MAX_METADATA_DEPTH = 6
MAX_METADATA_STRING_LENGTH = 1_000

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


__all__ = [
    "CONFIDENCE_LEVELS",
    "FORBIDDEN_CAPABILITY_METADATA_KEYS",
    "METRIC_SUPPORT_LEVELS",
    "CapabilityCandidate",
    "ProjectCapabilityFact",
    "normalize_capability_candidate",
    "normalize_project_capability_fact",
    "validate_capability_candidate",
    "validate_project_capability_fact",
]
