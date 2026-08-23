"""Strict, persistence-ready contracts for evidence-grounded engineering stories.

Engineering stories are bounded references over existing evidence authority.
This module performs no extraction, clustering, persistence, retrieval, ranking,
generation, or lifecycle transitions.  In particular, a story field is
claimable only when its own evidence state and provenance permit it.

Story ID generation is intentionally deferred until an authoritative event
clustering contract exists.  Callers must supply a stable semantic ID; mutable
story prose is never hashed into identity here.

ID validation establishes only a bounded authority-reference domain.  Before
any downstream claim use, referenced facts, capabilities, and boundaries must
still resolve in the exact same authoritative project memory, and lifecycle
and claim-boundary restrictions must still be applied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
import json
import re
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from backend.project_evidence_models import MAX_DETAIL_LENGTH
from backend.project_repository_identity import normalize_project_id


MAX_STORY_FIELD_VALUE_LENGTH = MAX_DETAIL_LENGTH
MAX_STORY_FIELD_PROVENANCE_IDS = 16
MAX_STORY_PROVENANCE_IDS = 64
MAX_STORY_SUFFICIENCY_FIELDS = 12
MAX_STORY_OPPORTUNITY_SIGNALS = 6

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,299}$")
_CLAIMABLE_PLACEHOLDERS = frozenset({
    "missing",
    "n/a",
    "none",
    "not known",
    "not sure",
    "plausible",
    "tbd",
    "unknown",
})
_UNSAFE_VALUE_PATTERNS = (
    re.compile(r"\bdiff --git\b", re.IGNORECASE),
    re.compile(r"@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|secret|credential)\s*=\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token)\s*:\s*['\"]?[a-z0-9+/=_-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:password|secret|credential)\s*:\s*(['\"])[^'\"]{4,}\1",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:password|secret|credential)\s*:\s*[a-z0-9+/=_-]{12,}[.;,]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bauthorization\s*:\s*(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(r"\bbearer\s+[a-z0-9._~+/=-]{8,}", re.IGNORECASE),
)
_M = TypeVar("_M", bound="EngineeringStoryContract")


class StoryFieldEvidenceState(str, Enum):
    CONFIRMED = "confirmed"
    SUPPORTED = "supported"
    PLAUSIBLE_MISSING = "plausible_missing"
    UNSUPPORTED = "unsupported"


class EngineeringStoryType(str, Enum):
    ARCHITECTURE_CHANGE = "architecture_change"
    RELIABILITY_HARDENING = "reliability_hardening"
    DEBUGGING_AND_REPAIR = "debugging_and_repair"
    RETRIEVAL_REDESIGN = "retrieval_redesign"
    VALIDATION_AND_QUALITY = "validation_and_quality"
    DATA_OR_MEMORY_SYSTEM = "data_or_memory_system"
    WORKFLOW_AUTOMATION = "workflow_automation"
    PERFORMANCE_OR_EFFICIENCY = "performance_or_efficiency"
    INTEGRATION = "integration"
    OTHER = "other"


class EngineeringStoryFieldName(str, Enum):
    PROBLEM_CONTEXT = "problem_context"
    TRIGGER = "trigger"
    BEFORE_STATE = "before_state"
    DECISION = "decision"
    MECHANISM = "mechanism"
    IMPLEMENTATION = "implementation"
    TRADEOFF = "tradeoff"
    VALIDATION = "validation"
    AFTER_STATE = "after_state"
    OBSERVABLE_OUTCOME = "observable_outcome"
    OWNERSHIP = "ownership"
    STAKEHOLDER_CONTEXT = "stakeholder_context"


class StoryContextGap(str, Enum):
    PROBLEM_CONTEXT = "problem_context"
    TRIGGER = "trigger"
    DECISION_REASON = "decision_reason"
    TRADEOFF = "tradeoff"
    OWNERSHIP = "ownership"
    STAKEHOLDER_CONTEXT = "stakeholder_context"
    OBSERVABLE_OUTCOME_CONTEXT = "observable_outcome_context"


class EngineeringStoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    STALE = "stale"
    CONFLICTED = "conflicted"


class SufficiencyLevel(str, Enum):
    UNASSESSED = "unassessed"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StoryOpportunityLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StoryOpportunitySignal(str, Enum):
    ARCHITECTURE_MIGRATION = "architecture_migration"
    DEFENSIVE_ENGINEERING_CLUSTER = "defensive_engineering_cluster"
    FAILURE_SPECIFIC_TEST_CLUSTER = "failure_specific_test_cluster"
    REPEATED_SUBSYSTEM_HARDENING = "repeated_subsystem_hardening"
    MAJOR_DESIGN_DECISION = "major_design_decision"
    MISSING_HUMAN_OR_WORKFLOW_CONTEXT = "missing_human_or_workflow_context"


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from exc


def _bounded_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > MAX_STORY_FIELD_VALUE_LENGTH:
        raise ValueError(
            f"{name} exceeds maximum length {MAX_STORY_FIELD_VALUE_LENGTH}"
        )
    if _CONTROL_RE.search(normalized):
        raise ValueError(f"{name} contains control characters")
    if any(pattern.search(normalized) for pattern in _UNSAFE_VALUE_PATTERNS):
        raise ValueError(f"{name} contains raw or sensitive source content")
    return normalized


def _safe_id(value: Any, name: str, *, prefix: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value != value.strip() or _CONTROL_RE.search(value):
        raise ValueError(f"{name} must be an exact normalized identifier")
    suffix = value[len(prefix):] if value.startswith(prefix) else ""
    if (
        not _SAFE_ID_RE.fullmatch(value)
        or not suffix
        or not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", suffix)
    ):
        raise ValueError(f"{name} must be a normalized {prefix} identifier")
    return value


def validate_engineering_story_id(value: str) -> str:
    """Validate a caller-supplied stable ID without inventing event identity."""

    return _safe_id(value, "story_id", prefix="engineering_story_")


def _stable_ids(
    values: Sequence[str],
    name: str,
    *,
    prefix: str,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    normalized = {_safe_id(value, name, prefix=prefix) for value in values}
    return tuple(sorted(normalized))


def _stable_enums(
    values: Sequence[Any],
    enum_type: type[Enum],
    name: str,
    *,
    maximum: int,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    index = {item: position for position, item in enumerate(enum_type)}
    normalized = {_enum_value(value, enum_type, name) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, EngineeringStoryContract):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported engineering story value: {type(value).__name__}")


def _decode(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        item_type = args[0]
        return tuple(_decode(item_type, item) for item in value)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [item for item in args if item is not type(None)]
        if len(choices) != 1:
            raise TypeError("unsupported union contract")
        return _decode(choices[0], value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, EngineeringStoryContract):
        if not isinstance(value, Mapping):
            raise TypeError("expected an object")
        return annotation.from_dict(value)
    return value


class EngineeringStoryContract:
    """Small strict serializer shared only by engineering-story contracts."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _serialize(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls: type[_M], payload: Mapping[str, Any]) -> _M:
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict expects an object")
        if any(not isinstance(key, str) for key in payload):
            raise TypeError(f"{cls.__name__} field names must be strings")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}"
            )
        hints = get_type_hints(cls)
        return cls(**{
            key: _decode(hints[key], value)
            for key, value in payload.items()
        })

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class EngineeringStoryField(EngineeringStoryContract):
    value: str | None
    evidence_state: StoryFieldEvidenceState
    evidence_fact_ids: tuple[str, ...] = ()
    capability_fact_ids: tuple[str, ...] = ()
    claim_boundary_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        state = _enum_value(
            self.evidence_state,
            StoryFieldEvidenceState,
            "evidence_state",
        )
        evidence_ids = _stable_ids(
            self.evidence_fact_ids,
            "evidence_fact_ids",
            prefix="pef_",
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        capability_ids = _stable_ids(
            self.capability_fact_ids,
            "capability_fact_ids",
            prefix="pcf_",
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        boundary_ids = _stable_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=MAX_STORY_FIELD_PROVENANCE_IDS,
        )
        if state in {
            StoryFieldEvidenceState.CONFIRMED,
            StoryFieldEvidenceState.SUPPORTED,
        }:
            value = _bounded_value(self.value, "value")
            if value.casefold() in _CLAIMABLE_PLACEHOLDERS:
                raise ValueError("claimable story fields require a concrete value")
            if not evidence_ids and not capability_ids:
                raise ValueError(
                    "confirmed or supported story fields require positive authority references"
                )
        else:
            if self.value is not None:
                raise ValueError(
                    "plausible_missing or unsupported story fields cannot carry a claimable value"
                )
            value = None
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "evidence_state", state)
        object.__setattr__(self, "evidence_fact_ids", evidence_ids)
        object.__setattr__(self, "capability_fact_ids", capability_ids)
        object.__setattr__(self, "claim_boundary_ids", boundary_ids)

    @property
    def has_positive_value(self) -> bool:
        """Return structural positivity, not final claim-boundary approval."""

        return self.evidence_state in {
            StoryFieldEvidenceState.CONFIRMED,
            StoryFieldEvidenceState.SUPPORTED,
        }


@dataclass(frozen=True, slots=True)
class EngineeringStoryLifecycle(EngineeringStoryContract):
    status: EngineeringStoryStatus
    requires_revalidation: bool = False
    superseded_by_story_id: str | None = None

    def __post_init__(self) -> None:
        status = _enum_value(self.status, EngineeringStoryStatus, "status")
        if not isinstance(self.requires_revalidation, bool):
            raise TypeError("requires_revalidation must be a boolean")
        successor = self.superseded_by_story_id
        if successor is not None:
            successor = _safe_id(
                successor,
                "superseded_by_story_id",
                prefix="engineering_story_",
            )
        if status is EngineeringStoryStatus.SUPERSEDED and successor is None:
            raise ValueError("superseded stories require a successor story ID")
        if status is not EngineeringStoryStatus.SUPERSEDED and successor is not None:
            raise ValueError("only superseded stories may identify a successor")
        if status in {
            EngineeringStoryStatus.STALE,
            EngineeringStoryStatus.CONFLICTED,
        } and not self.requires_revalidation:
            raise ValueError("stale or conflicted stories require revalidation")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "superseded_by_story_id", successor)


@dataclass(frozen=True, slots=True)
class ClaimSufficiency(EngineeringStoryContract):
    level: SufficiencyLevel
    supported_fields: tuple[EngineeringStoryFieldName, ...] = ()
    missing_fields: tuple[EngineeringStoryFieldName, ...] = ()

    def __post_init__(self) -> None:
        _normalize_sufficiency(self, "claim_sufficiency")


@dataclass(frozen=True, slots=True)
class StorySufficiency(EngineeringStoryContract):
    level: SufficiencyLevel
    supported_fields: tuple[EngineeringStoryFieldName, ...] = ()
    missing_fields: tuple[EngineeringStoryFieldName, ...] = ()

    def __post_init__(self) -> None:
        _normalize_sufficiency(self, "story_sufficiency")


def _normalize_sufficiency(
    assessment: ClaimSufficiency | StorySufficiency,
    name: str,
) -> None:
    level = _enum_value(assessment.level, SufficiencyLevel, f"{name}.level")
    supported = _stable_enums(
        assessment.supported_fields,
        EngineeringStoryFieldName,
        f"{name}.supported_fields",
        maximum=MAX_STORY_SUFFICIENCY_FIELDS,
    )
    missing = _stable_enums(
        assessment.missing_fields,
        EngineeringStoryFieldName,
        f"{name}.missing_fields",
        maximum=MAX_STORY_SUFFICIENCY_FIELDS,
    )
    if set(supported) & set(missing):
        raise ValueError(f"{name} cannot mark a field both supported and missing")
    if level is SufficiencyLevel.UNASSESSED and (supported or missing):
        raise ValueError(f"unassessed {name} cannot contain assessed fields")
    if level is SufficiencyLevel.HIGH and not supported:
        raise ValueError(f"high {name} requires at least one supported field")
    object.__setattr__(assessment, "level", level)
    object.__setattr__(assessment, "supported_fields", supported)
    object.__setattr__(assessment, "missing_fields", missing)


@dataclass(frozen=True, slots=True)
class StoryOpportunity(EngineeringStoryContract):
    level: StoryOpportunityLevel
    signals: tuple[StoryOpportunitySignal, ...] = ()
    missing_context: tuple[StoryContextGap, ...] = ()

    def __post_init__(self) -> None:
        level = _enum_value(self.level, StoryOpportunityLevel, "level")
        signals = _stable_enums(
            self.signals,
            StoryOpportunitySignal,
            "signals",
            maximum=MAX_STORY_OPPORTUNITY_SIGNALS,
        )
        missing = _stable_enums(
            self.missing_context,
            StoryContextGap,
            "missing_context",
            maximum=MAX_STORY_SUFFICIENCY_FIELDS,
        )
        if level is StoryOpportunityLevel.NONE and (signals or missing):
            raise ValueError("no story opportunity cannot carry signals or missing context")
        if level is not StoryOpportunityLevel.NONE and not (signals or missing):
            raise ValueError("a story opportunity requires a signal or missing context")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "missing_context", missing)


_STORY_FIELD_ATTRIBUTES: tuple[tuple[EngineeringStoryFieldName, str], ...] = (
    (EngineeringStoryFieldName.PROBLEM_CONTEXT, "problem_context"),
    (EngineeringStoryFieldName.TRIGGER, "trigger"),
    (EngineeringStoryFieldName.BEFORE_STATE, "before_state"),
    (EngineeringStoryFieldName.DECISION, "decision"),
    (EngineeringStoryFieldName.MECHANISM, "mechanism"),
    (EngineeringStoryFieldName.IMPLEMENTATION, "implementation"),
    (EngineeringStoryFieldName.TRADEOFF, "tradeoff"),
    (EngineeringStoryFieldName.VALIDATION, "validation"),
    (EngineeringStoryFieldName.AFTER_STATE, "after_state"),
    (EngineeringStoryFieldName.OBSERVABLE_OUTCOME, "observable_outcome"),
    (EngineeringStoryFieldName.OWNERSHIP, "ownership"),
    (EngineeringStoryFieldName.STAKEHOLDER_CONTEXT, "stakeholder_context"),
)


@dataclass(frozen=True, slots=True)
class EngineeringStory(EngineeringStoryContract):
    story_id: str
    project_id: str
    story_type: EngineeringStoryType
    problem_context: EngineeringStoryField
    trigger: EngineeringStoryField
    before_state: EngineeringStoryField
    decision: EngineeringStoryField
    mechanism: EngineeringStoryField
    implementation: EngineeringStoryField
    tradeoff: EngineeringStoryField
    validation: EngineeringStoryField
    after_state: EngineeringStoryField
    observable_outcome: EngineeringStoryField
    ownership: EngineeringStoryField
    stakeholder_context: EngineeringStoryField
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    claim_boundary_ids: tuple[str, ...]
    lifecycle: EngineeringStoryLifecycle
    claim_sufficiency: ClaimSufficiency
    story_sufficiency: StorySufficiency
    opportunity: StoryOpportunity

    def __post_init__(self) -> None:
        story_id = validate_engineering_story_id(self.story_id)
        project_id = normalize_project_id(self.project_id)
        if not project_id or project_id != self.project_id:
            raise ValueError("project_id must be an exact canonical project identifier")
        story_type = _enum_value(self.story_type, EngineeringStoryType, "story_type")
        story_fields: dict[EngineeringStoryFieldName, EngineeringStoryField] = {}
        for field_name, attribute in _STORY_FIELD_ATTRIBUTES:
            value = getattr(self, attribute)
            if not isinstance(value, EngineeringStoryField):
                raise TypeError(f"{attribute} must be an EngineeringStoryField")
            story_fields[field_name] = value
        evidence_ids = _stable_ids(
            self.evidence_fact_ids,
            "evidence_fact_ids",
            prefix="pef_",
            maximum=MAX_STORY_PROVENANCE_IDS,
        )
        capability_ids = _stable_ids(
            self.capability_fact_ids,
            "capability_fact_ids",
            prefix="pcf_",
            maximum=MAX_STORY_PROVENANCE_IDS,
        )
        boundary_ids = _stable_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=MAX_STORY_PROVENANCE_IDS,
        )
        if not evidence_ids and not capability_ids:
            raise ValueError("an engineering story requires positive authority references")
        if not any(field.has_positive_value for field in story_fields.values()):
            raise ValueError("an engineering story requires at least one positive story field")
        for field in story_fields.values():
            if not set(field.evidence_fact_ids).issubset(evidence_ids):
                raise ValueError("field evidence IDs must exist in story provenance")
            if not set(field.capability_fact_ids).issubset(capability_ids):
                raise ValueError("field capability IDs must exist in story provenance")
            if not set(field.claim_boundary_ids).issubset(boundary_ids):
                raise ValueError("field claim-boundary IDs must exist in story provenance")
        if not isinstance(self.lifecycle, EngineeringStoryLifecycle):
            raise TypeError("lifecycle must be an EngineeringStoryLifecycle")
        if self.lifecycle.superseded_by_story_id == story_id:
            raise ValueError("an engineering story cannot supersede itself")
        if not isinstance(self.claim_sufficiency, ClaimSufficiency):
            raise TypeError("claim_sufficiency must be a ClaimSufficiency")
        if not isinstance(self.story_sufficiency, StorySufficiency):
            raise TypeError("story_sufficiency must be a StorySufficiency")
        if not isinstance(self.opportunity, StoryOpportunity):
            raise TypeError("opportunity must be a StoryOpportunity")
        _validate_assessment_fields(
            self.claim_sufficiency,
            story_fields,
            "claim_sufficiency",
        )
        _validate_assessment_fields(
            self.story_sufficiency,
            story_fields,
            "story_sufficiency",
        )
        object.__setattr__(self, "story_id", story_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "story_type", story_type)
        object.__setattr__(self, "evidence_fact_ids", evidence_ids)
        object.__setattr__(self, "capability_fact_ids", capability_ids)
        object.__setattr__(self, "claim_boundary_ids", boundary_ids)


def _validate_assessment_fields(
    assessment: ClaimSufficiency | StorySufficiency,
    story_fields: Mapping[EngineeringStoryFieldName, EngineeringStoryField],
    name: str,
) -> None:
    if any(not story_fields[field_name].has_positive_value for field_name in assessment.supported_fields):
        raise ValueError(f"{name} supported fields must have positive evidence state")
    if any(story_fields[field_name].has_positive_value for field_name in assessment.missing_fields):
        raise ValueError(f"{name} missing fields must not have positive evidence state")


__all__ = [
    "ClaimSufficiency",
    "EngineeringStory",
    "EngineeringStoryField",
    "EngineeringStoryFieldName",
    "EngineeringStoryLifecycle",
    "EngineeringStoryStatus",
    "EngineeringStoryType",
    "MAX_STORY_FIELD_PROVENANCE_IDS",
    "MAX_STORY_FIELD_VALUE_LENGTH",
    "MAX_STORY_OPPORTUNITY_SIGNALS",
    "MAX_STORY_PROVENANCE_IDS",
    "MAX_STORY_SUFFICIENCY_FIELDS",
    "StoryFieldEvidenceState",
    "StoryContextGap",
    "StoryOpportunity",
    "StoryOpportunityLevel",
    "StoryOpportunitySignal",
    "StorySufficiency",
    "SufficiencyLevel",
    "validate_engineering_story_id",
]
