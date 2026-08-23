"""Pure authority resolution and structural inputs for future story clustering.

This module resolves existing project Evidence Facts, Capability Facts, Claim
Boundaries, and source references.  Its outputs are bounded derived views, not
replacement evidence authority and not reconstructed Engineering Stories.

Event anchors are stable structural tuples.  They intentionally have no hash
ID and are not Engineering Story IDs; story membership and story identity stay
deferred until a later clustering step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import re
from typing import Any, TypeVar

from backend.engineering_story_models import EngineeringStoryContract
from backend.project_claim_boundaries import validate_project_claim_boundary
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
    ProjectEvidenceMemory,
    build_project_evidence_stable_id,
)
from backend.project_repository_identity import normalize_project_id


MAX_AUTHORITY_RECORDS = 5_000
MAX_STORY_EVIDENCE_INPUTS = 32
MAX_STORY_CAPABILITY_LINEAGES = 16
MAX_STORY_CLAIM_BOUNDARIES = 64
MAX_SOURCE_LINEAGES_PER_INPUT = 16
MAX_EVENT_ANCHORS = 128
MAX_STORY_EVIDENCE_RELATIONS = 256
MAX_TECHNICAL_TAGS = 32
MAX_RELATION_BASIS_IDS = 4

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_AUTHORITY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,299}$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,299}$")
_SOURCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:/|/|//)")
_SENSITIVE_RE = re.compile(
    r"(?:diff --git|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:api[_ -]?key|access[_ -]?token|password|secret|credential)\s*=|"
    r"\bauthorization\s*:\s*(?:bearer|basic)\b|\bbearer\s+[a-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
_A = TypeVar("_A", ProjectEvidenceFact, ProjectCapabilityFact, ProjectClaimBoundary)


class SourceLineageState(str, Enum):
    AVAILABLE = "available"
    MISSING_STRUCTURAL_CONTEXT = "missing_structural_context"


class CapabilityLineageState(str, Enum):
    RESOLVED = "resolved"
    NOT_PRESENT = "not_present"


class StoryEvidenceLineageState(str, Enum):
    COMPLETE = "complete"
    MISSING_STRUCTURAL_CONTEXT = "missing_structural_context"


class StoryEventAnchorKind(str, Enum):
    EXPLICIT_CHANGE = "explicit_change"
    PARENT_CHANGE = "parent_change"
    COMMIT_AND_SYMBOL = "commit_and_symbol"
    COMMIT_AND_PATH = "commit_and_path"
    SOURCE_IDENTITY = "source_identity"


class StoryEvidenceRelationType(str, Enum):
    SAME_EXPLICIT_CHANGE = "same_explicit_change"
    SAME_PARENT_CHANGE = "same_parent_change"
    SAME_COMMIT = "same_commit"
    SAME_SOURCE = "same_source"
    SAME_PATH = "same_path"
    SAME_SYMBOL = "same_symbol"
    CAPABILITY_SUPPORT_RELATION = "capability_support_relation"
    SHARED_VALIDATED_SOURCE_LINEAGE = "shared_validated_source_lineage"


class StoryEvidenceRelationStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class StoryEvidenceResolutionCode(str, Enum):
    INVALID_PROJECT_ID = "invalid_project_id"
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    MALFORMED_AUTHORITY_ID = "malformed_authority_id"
    CROSS_PROJECT_AUTHORITY = "cross_project_authority"
    CONFLICTING_EVIDENCE_FACT = "conflicting_evidence_fact"
    CONFLICTING_CAPABILITY = "conflicting_capability"
    CONFLICTING_CLAIM_BOUNDARY = "conflicting_claim_boundary"
    UNKNOWN_EVIDENCE_FACT = "unknown_evidence_fact"
    UNKNOWN_CAPABILITY = "unknown_capability"
    UNKNOWN_CLAIM_BOUNDARY = "unknown_claim_boundary"
    INVALID_CAPABILITY_LINEAGE = "invalid_capability_lineage"
    INVALID_CLAIM_BOUNDARY = "invalid_claim_boundary"
    IRRELEVANT_CLAIM_BOUNDARY = "irrelevant_claim_boundary"
    MALFORMED_SOURCE_LINEAGE = "malformed_source_lineage"
    CONFLICTING_SOURCE_LINEAGE = "conflicting_source_lineage"
    NO_EVIDENCE_INPUTS = "no_evidence_inputs"


class StoryEvidenceResolutionError(ValueError):
    """Bounded deterministic failure from story-evidence authority resolution."""

    def __init__(
        self,
        code: StoryEvidenceResolutionCode | str,
        reference_id: str | None = None,
    ) -> None:
        self.code = StoryEvidenceResolutionCode(code)
        self.reference_id = _diagnostic_id(reference_id)
        message = self.code.value
        if self.reference_id is not None:
            message += f":{self.reference_id}"
        super().__init__(message)


def _fail(
    code: StoryEvidenceResolutionCode,
    reference_id: str | None = None,
) -> None:
    raise StoryEvidenceResolutionError(code, reference_id)


def _diagnostic_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > 100 or _CONTROL_RE.search(normalized):
        return None
    return normalized if _SOURCE_ID_RE.fullmatch(normalized) else None


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        _fail(StoryEvidenceResolutionCode.INVALID_PROJECT_ID)
    return normalized


def _authority_id(value: Any, name: str, prefix: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    suffix = value[len(prefix):] if value.startswith(prefix) else ""
    if (
        value != value.strip()
        or not suffix
        or not _AUTHORITY_ID_RE.fullmatch(value)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", suffix)
    ):
        _fail(StoryEvidenceResolutionCode.MALFORMED_AUTHORITY_ID, value)
    return value


def _stable_authority_ids(
    values: Sequence[str],
    name: str,
    *,
    prefix: str,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    return tuple(sorted({_authority_id(value, name, prefix) for value in values}))


def _bounded_text(
    value: Any,
    name: str,
    maximum: int,
    *,
    required: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value != value.strip() or _CONTROL_RE.search(value):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    if required and not value:
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    if len(value) > maximum or _SENSITIVE_RE.search(value):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return value


def _optional_text(value: Any, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = _bounded_text(value, name, maximum)
    return normalized or None


def _safe_path(value: Any) -> str | None:
    path = _optional_text(value, "file_path", 500)
    if path is None:
        return None
    normalized = path.replace("\\", "/")
    if (
        _ABSOLUTE_PATH_RE.match(normalized)
        or any(part == ".." for part in normalized.split("/"))
    ):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return normalized


def _safe_commit(value: Any) -> str | None:
    commit = _optional_text(value, "commit_sha", 64)
    if commit is None:
        return None
    if not _COMMIT_RE.fullmatch(commit):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return commit.lower()


def _metadata_id(metadata: Mapping[str, Any], key: str) -> str | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if not isinstance(value, str) or not _SOURCE_ID_RE.fullmatch(value):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return value


def _source_type(value: Any) -> str:
    source_type = _bounded_text(value, "source_type", 100, required=True)
    if not _SOURCE_TYPE_RE.fullmatch(source_type):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return source_type


def _source_id(value: Any) -> str:
    source_id = _bounded_text(value, "source_id", 300, required=True)
    if not _SOURCE_ID_RE.fullmatch(source_id):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE)
    return source_id


def _stable_tags(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("technical_tags must be a sequence")
    if len(values) > MAX_TECHNICAL_TAGS:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    tags: set[str] = set()
    for value in values:
        tag = _bounded_text(value, "technical_tag", 100, required=True)
        tags.add(tag)
    return tuple(sorted(tags, key=lambda item: (item.casefold(), item)))


@dataclass(frozen=True, slots=True)
class StorySourceLineage(EngineeringStoryContract):
    project_id: str
    source_type: str
    source_id: str
    content_hash: str
    state: SourceLineageState
    repository: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    symbol: str | None = None
    upstream_source_id: str | None = None
    explicit_change_id: str | None = None
    parent_change_id: str | None = None
    parent_source_id: str | None = None

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        source_type = _source_type(self.source_type)
        source_id = _source_id(self.source_id)
        if not isinstance(self.content_hash, str) or not _SHA256_RE.fullmatch(
            self.content_hash
        ):
            _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE, source_id)
        repository = _optional_text(self.repository, "repository", 300)
        commit_sha = _safe_commit(self.commit_sha)
        file_path = _safe_path(self.file_path)
        symbol = _optional_text(self.symbol, "symbol", 300)
        optional_ids = {}
        for name in (
            "upstream_source_id",
            "explicit_change_id",
            "parent_change_id",
            "parent_source_id",
        ):
            value = getattr(self, name)
            if value is not None and not _SOURCE_ID_RE.fullmatch(value):
                _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE, source_id)
            optional_ids[name] = value
        has_context = any((
            repository,
            commit_sha,
            file_path,
            symbol,
            *optional_ids.values(),
        ))
        state = SourceLineageState(self.state)
        expected = (
            SourceLineageState.AVAILABLE
            if has_context
            else SourceLineageState.MISSING_STRUCTURAL_CONTEXT
        )
        if state is not expected:
            _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE, source_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "commit_sha", commit_sha)
        object.__setattr__(self, "file_path", file_path)
        object.__setattr__(self, "symbol", symbol)
        for name, value in optional_ids.items():
            object.__setattr__(self, name, value)

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return (self.project_id, self.source_type, self.source_id)


_ANCHOR_STRENGTH = {
    StoryEventAnchorKind.EXPLICIT_CHANGE: StoryEvidenceRelationStrength.STRONG,
    StoryEventAnchorKind.PARENT_CHANGE: StoryEvidenceRelationStrength.MODERATE,
    StoryEventAnchorKind.COMMIT_AND_SYMBOL: StoryEvidenceRelationStrength.STRONG,
    StoryEventAnchorKind.COMMIT_AND_PATH: StoryEvidenceRelationStrength.MODERATE,
    StoryEventAnchorKind.SOURCE_IDENTITY: StoryEvidenceRelationStrength.WEAK,
}


@dataclass(frozen=True, slots=True)
class StoryEventAnchor(EngineeringStoryContract):
    project_id: str
    anchor_kind: StoryEventAnchorKind
    strength: StoryEvidenceRelationStrength
    explicit_change_id: str | None = None
    parent_change_id: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    symbol: str | None = None
    source_type: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        kind = StoryEventAnchorKind(self.anchor_kind)
        strength = StoryEvidenceRelationStrength(self.strength)
        if strength is not _ANCHOR_STRENGTH[kind]:
            raise ValueError("event anchor strength conflicts with anchor kind")
        populated = {
            "explicit_change_id": self.explicit_change_id,
            "parent_change_id": self.parent_change_id,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "file_path": self.file_path,
            "symbol": self.symbol,
            "source_type": self.source_type,
            "source_id": self.source_id,
        }
        required_by_kind = {
            StoryEventAnchorKind.EXPLICIT_CHANGE: {"explicit_change_id"},
            StoryEventAnchorKind.PARENT_CHANGE: {"parent_change_id"},
            StoryEventAnchorKind.COMMIT_AND_SYMBOL: {
                "repository", "commit_sha", "symbol",
            },
            StoryEventAnchorKind.COMMIT_AND_PATH: {
                "repository", "commit_sha", "file_path",
            },
            StoryEventAnchorKind.SOURCE_IDENTITY: {"source_type", "source_id"},
        }[kind]
        if {name for name, value in populated.items() if value is not None} != required_by_kind:
            raise ValueError("event anchor contains invalid structural dimensions")
        if self.explicit_change_id is not None:
            _source_id(self.explicit_change_id)
        if self.parent_change_id is not None:
            _source_id(self.parent_change_id)
        if self.repository is not None:
            _bounded_text(self.repository, "repository", 300, required=True)
        if self.commit_sha is not None:
            object.__setattr__(self, "commit_sha", _safe_commit(self.commit_sha))
        if self.file_path is not None:
            object.__setattr__(self, "file_path", _safe_path(self.file_path))
        if self.symbol is not None:
            _bounded_text(self.symbol, "symbol", 300, required=True)
        if self.source_type is not None:
            _source_type(self.source_type)
        if self.source_id is not None:
            _source_id(self.source_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "anchor_kind", kind)
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True, slots=True)
class CapabilityEvidenceLineage(EngineeringStoryContract):
    project_id: str
    capability_id: str
    capability_type: str
    source_evidence_fact_ids: tuple[str, ...]
    present: bool
    confidence: Confidence
    metric_support: MetricSupport
    state: CapabilityLineageState

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        capability_id = _authority_id(self.capability_id, "capability_id", "pcf_")
        capability_type = _bounded_text(
            self.capability_type, "capability_type", 100, required=True
        )
        evidence_ids = _stable_authority_ids(
            self.source_evidence_fact_ids,
            "source_evidence_fact_ids",
            prefix="pef_",
            maximum=MAX_STORY_EVIDENCE_INPUTS,
        )
        if not isinstance(self.present, bool):
            raise TypeError("present must be a boolean")
        confidence = Confidence(self.confidence)
        metric_support = MetricSupport(self.metric_support)
        state = CapabilityLineageState(self.state)
        expected = (
            CapabilityLineageState.RESOLVED
            if self.present
            else CapabilityLineageState.NOT_PRESENT
        )
        if state is not expected or (self.present and not evidence_ids):
            _fail(StoryEvidenceResolutionCode.INVALID_CAPABILITY_LINEAGE, capability_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "capability_type", capability_type)
        object.__setattr__(self, "source_evidence_fact_ids", evidence_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metric_support", metric_support)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True, slots=True)
class StoryEvidenceInput(EngineeringStoryContract):
    project_id: str
    evidence_fact_id: str
    evidence_type: EvidenceType
    evidence_status: EvidenceStatus
    confidence: Confidence
    metric_support: MetricSupport
    technical_tags: tuple[str, ...]
    source_lineages: tuple[StorySourceLineage, ...]
    event_anchors: tuple[StoryEventAnchor, ...]
    capability_ids: tuple[str, ...] = ()
    claim_boundary_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        evidence_id = _authority_id(
            self.evidence_fact_id, "evidence_fact_id", "pef_"
        )
        evidence_type = EvidenceType(self.evidence_type)
        evidence_status = EvidenceStatus(self.evidence_status)
        confidence = Confidence(self.confidence)
        metric_support = MetricSupport(self.metric_support)
        tags = _stable_tags(self.technical_tags)
        lineages = _stable_lineages(self.source_lineages, project_id=project_id)
        if not lineages:
            _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE, evidence_id)
        anchors = _stable_anchors(self.event_anchors, project_id=project_id)
        if not anchors:
            raise ValueError("story evidence input requires at least one event anchor")
        capability_ids = _stable_authority_ids(
            self.capability_ids,
            "capability_ids",
            prefix="pcf_",
            maximum=MAX_STORY_CAPABILITY_LINEAGES,
        )
        boundary_ids = _stable_authority_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=MAX_STORY_CLAIM_BOUNDARIES,
        )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "evidence_fact_id", evidence_id)
        object.__setattr__(self, "evidence_type", evidence_type)
        object.__setattr__(self, "evidence_status", evidence_status)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metric_support", metric_support)
        object.__setattr__(self, "technical_tags", tags)
        object.__setattr__(self, "source_lineages", lineages)
        object.__setattr__(self, "event_anchors", anchors)
        object.__setattr__(self, "capability_ids", capability_ids)
        object.__setattr__(self, "claim_boundary_ids", boundary_ids)


_RELATION_STRENGTH = {
    StoryEvidenceRelationType.SAME_EXPLICIT_CHANGE: StoryEvidenceRelationStrength.STRONG,
    StoryEvidenceRelationType.SAME_PARENT_CHANGE: StoryEvidenceRelationStrength.MODERATE,
    StoryEvidenceRelationType.SAME_COMMIT: StoryEvidenceRelationStrength.STRONG,
    StoryEvidenceRelationType.SAME_SOURCE: StoryEvidenceRelationStrength.STRONG,
    StoryEvidenceRelationType.SAME_PATH: StoryEvidenceRelationStrength.MODERATE,
    StoryEvidenceRelationType.SAME_SYMBOL: StoryEvidenceRelationStrength.WEAK,
    StoryEvidenceRelationType.CAPABILITY_SUPPORT_RELATION: StoryEvidenceRelationStrength.MODERATE,
    StoryEvidenceRelationType.SHARED_VALIDATED_SOURCE_LINEAGE: StoryEvidenceRelationStrength.MODERATE,
}


@dataclass(frozen=True, slots=True)
class StoryEvidenceRelation(EngineeringStoryContract):
    project_id: str
    left_evidence_fact_id: str
    right_evidence_fact_id: str
    relation_type: StoryEvidenceRelationType
    strength: StoryEvidenceRelationStrength
    basis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        left = _authority_id(
            self.left_evidence_fact_id, "left_evidence_fact_id", "pef_"
        )
        right = _authority_id(
            self.right_evidence_fact_id, "right_evidence_fact_id", "pef_"
        )
        if left >= right:
            raise ValueError("relation evidence IDs must be distinct and ordered")
        relation_type = StoryEvidenceRelationType(self.relation_type)
        strength = StoryEvidenceRelationStrength(self.strength)
        if strength is not _RELATION_STRENGTH[relation_type]:
            raise ValueError("relation strength conflicts with relation type")
        if isinstance(self.basis_ids, (str, bytes)) or not isinstance(
            self.basis_ids, Sequence
        ):
            raise TypeError("basis_ids must be a sequence")
        if not self.basis_ids or len(self.basis_ids) > MAX_RELATION_BASIS_IDS:
            _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
        basis = tuple(sorted({
            _bounded_text(value, "basis_id", 500, required=True)
            for value in self.basis_ids
        }))
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "left_evidence_fact_id", left)
        object.__setattr__(self, "right_evidence_fact_id", right)
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "basis_ids", basis)


@dataclass(frozen=True, slots=True)
class StoryEvidenceBundle(EngineeringStoryContract):
    project_id: str
    evidence_inputs: tuple[StoryEvidenceInput, ...]
    capability_lineages: tuple[CapabilityEvidenceLineage, ...]
    claim_boundary_ids: tuple[str, ...]
    event_anchors: tuple[StoryEventAnchor, ...]
    relations: tuple[StoryEvidenceRelation, ...]
    lineage_state: StoryEvidenceLineageState

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        if not self.evidence_inputs:
            _fail(StoryEvidenceResolutionCode.NO_EVIDENCE_INPUTS)
        if len(self.evidence_inputs) > MAX_STORY_EVIDENCE_INPUTS:
            _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
        if isinstance(self.evidence_inputs, (str, bytes)) or not isinstance(
            self.evidence_inputs, Sequence
        ):
            raise TypeError("evidence_inputs must be a sequence")
        if any(not isinstance(item, StoryEvidenceInput) for item in self.evidence_inputs):
            raise TypeError("evidence_inputs must contain StoryEvidenceInput values")
        inputs = tuple(sorted(self.evidence_inputs, key=_evidence_input_sort_key))
        if any(item.project_id != project_id for item in inputs):
            _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
        evidence_ids = [item.evidence_fact_id for item in inputs]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("story evidence bundle contains duplicate evidence inputs")
        if len(self.capability_lineages) > MAX_STORY_CAPABILITY_LINEAGES:
            _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
        if isinstance(self.capability_lineages, (str, bytes)) or not isinstance(
            self.capability_lineages, Sequence
        ):
            raise TypeError("capability_lineages must be a sequence")
        if any(
            not isinstance(item, CapabilityEvidenceLineage)
            for item in self.capability_lineages
        ):
            raise TypeError(
                "capability_lineages must contain CapabilityEvidenceLineage values"
            )
        capabilities = tuple(sorted(
            self.capability_lineages,
            key=lambda item: item.capability_id,
        ))
        if any(item.project_id != project_id for item in capabilities):
            _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
        capability_ids = [item.capability_id for item in capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("story evidence bundle contains duplicate capabilities")
        boundary_ids = _stable_authority_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=MAX_STORY_CLAIM_BOUNDARIES,
        )
        anchors = _stable_anchors(self.event_anchors, project_id=project_id)
        expected_anchors = _stable_anchors(
            tuple(anchor for item in inputs for anchor in item.event_anchors),
            project_id=project_id,
        )
        if anchors != expected_anchors:
            raise ValueError("bundle event anchors must equal evidence input anchors")
        if len(self.relations) > MAX_STORY_EVIDENCE_RELATIONS:
            _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
        if isinstance(self.relations, (str, bytes)) or not isinstance(
            self.relations, Sequence
        ):
            raise TypeError("relations must be a sequence")
        if any(not isinstance(item, StoryEvidenceRelation) for item in self.relations):
            raise TypeError("relations must contain StoryEvidenceRelation values")
        relations = tuple(sorted(set(self.relations), key=_relation_sort_key))
        valid_ids = set(evidence_ids)
        if any(
            item.project_id != project_id
            or item.left_evidence_fact_id not in valid_ids
            or item.right_evidence_fact_id not in valid_ids
            for item in relations
        ):
            _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
        resolved_capability_ids = {
            item.capability_id
            for item in capabilities
            if item.state is CapabilityLineageState.RESOLVED
        }
        if any(
            not set(item.capability_ids).issubset(resolved_capability_ids)
            for item in inputs
        ):
            _fail(StoryEvidenceResolutionCode.INVALID_CAPABILITY_LINEAGE)
        if any(
            not set(item.claim_boundary_ids).issubset(boundary_ids)
            for item in inputs
        ):
            _fail(StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY)
        lineage_state = StoryEvidenceLineageState(self.lineage_state)
        expected_state = (
            StoryEvidenceLineageState.MISSING_STRUCTURAL_CONTEXT
            if any(
                lineage.state is SourceLineageState.MISSING_STRUCTURAL_CONTEXT
                for item in inputs
                for lineage in item.source_lineages
            )
            else StoryEvidenceLineageState.COMPLETE
        )
        if lineage_state is not expected_state:
            raise ValueError("bundle lineage state does not match source lineage")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "evidence_inputs", inputs)
        object.__setattr__(self, "capability_lineages", capabilities)
        object.__setattr__(self, "claim_boundary_ids", boundary_ids)
        object.__setattr__(self, "event_anchors", anchors)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "lineage_state", lineage_state)


def _lineage_sort_key(lineage: StorySourceLineage) -> tuple[str, ...]:
    return (
        lineage.explicit_change_id or "",
        lineage.parent_change_id or "",
        lineage.repository or "",
        lineage.commit_sha or "",
        lineage.file_path or "",
        lineage.symbol or "",
        lineage.source_type,
        lineage.source_id,
        lineage.content_hash,
    )


def _anchor_sort_key(anchor: StoryEventAnchor) -> tuple[Any, ...]:
    return (
        tuple(StoryEventAnchorKind).index(anchor.anchor_kind),
        anchor.explicit_change_id or "",
        anchor.parent_change_id or "",
        anchor.repository or "",
        anchor.commit_sha or "",
        anchor.file_path or "",
        anchor.symbol or "",
        anchor.source_type or "",
        anchor.source_id or "",
    )


def _evidence_input_sort_key(item: StoryEvidenceInput) -> tuple[Any, ...]:
    anchor_key = _anchor_sort_key(item.event_anchors[0]) if item.event_anchors else ()
    return (anchor_key, item.evidence_fact_id)


def _relation_sort_key(relation: StoryEvidenceRelation) -> tuple[Any, ...]:
    return (
        relation.left_evidence_fact_id,
        relation.right_evidence_fact_id,
        tuple(StoryEvidenceRelationType).index(relation.relation_type),
        relation.basis_ids,
    )


def _stable_lineages(
    values: Sequence[StorySourceLineage],
    *,
    project_id: str,
) -> tuple[StorySourceLineage, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("source_lineages must be a sequence")
    if len(values) > MAX_SOURCE_LINEAGES_PER_INPUT:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    if any(not isinstance(item, StorySourceLineage) for item in values):
        raise TypeError("source_lineages must contain StorySourceLineage values")
    if any(item.project_id != project_id for item in values):
        _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
    by_identity: dict[tuple[str, str, str], StorySourceLineage] = {}
    for item in values:
        previous = by_identity.get(item.identity_key)
        if previous is not None and previous != item:
            _fail(StoryEvidenceResolutionCode.CONFLICTING_SOURCE_LINEAGE, item.source_id)
        by_identity[item.identity_key] = item
    return tuple(sorted(by_identity.values(), key=_lineage_sort_key))


def _stable_anchors(
    values: Sequence[StoryEventAnchor],
    *,
    project_id: str,
) -> tuple[StoryEventAnchor, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("event_anchors must be a sequence")
    if len(values) > MAX_EVENT_ANCHORS:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    if any(not isinstance(item, StoryEventAnchor) for item in values):
        raise TypeError("event_anchors must contain StoryEventAnchor values")
    if any(item.project_id != project_id for item in values):
        _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
    return tuple(sorted(set(values), key=_anchor_sort_key))


def _authority_index(
    values: Sequence[_A],
    *,
    expected_type: type[_A],
    id_attribute: str,
    id_prefix: str,
    project_id: str,
    conflict_code: StoryEvidenceResolutionCode,
) -> dict[str, _A]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("authority records must be a sequence")
    if len(values) > MAX_AUTHORITY_RECORDS:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    output: dict[str, _A] = {}
    payloads: dict[str, str] = {}
    for item in values:
        if not isinstance(item, expected_type):
            raise TypeError(f"authority records must contain {expected_type.__name__}")
        item_id = _authority_id(
            getattr(item, id_attribute), id_attribute, id_prefix
        )
        if item.project_id != project_id:
            _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY, item_id)
        payload = item.to_json()
        previous = payloads.get(item_id)
        if previous is not None and previous != payload:
            _fail(conflict_code, item_id)
        payloads[item_id] = payload
        output[item_id] = item
    return output


def _lineage_from_source_ref(
    ref: EvidenceSourceRef,
    *,
    project_id: str,
) -> StorySourceLineage:
    if not isinstance(ref, EvidenceSourceRef) or ref.project_id != project_id:
        _fail(StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY)
    metadata = ref.metadata
    if not isinstance(metadata, Mapping):
        _fail(StoryEvidenceResolutionCode.MALFORMED_SOURCE_LINEAGE, ref.source_id)
    source_type = _source_type(ref.source_type)
    source_id = _source_id(ref.source_id)
    metadata_change = _metadata_id(metadata, "change_id")
    explicit_change_id = (
        source_id
        if source_type == "project_change_raw_change_summary"
        else metadata_change
    )
    if (
        source_type == "project_change_raw_change_summary"
        and metadata_change is not None
        and metadata_change != source_id
    ):
        _fail(StoryEvidenceResolutionCode.CONFLICTING_SOURCE_LINEAGE, source_id)
    values = {
        "repository": _optional_text(ref.repo, "repository", 300),
        "commit_sha": _safe_commit(ref.commit_sha),
        "file_path": _safe_path(ref.file_path),
        "symbol": _optional_text(ref.symbol, "symbol", 300),
        "upstream_source_id": _metadata_id(metadata, "upstream_source_id"),
        "explicit_change_id": explicit_change_id,
        "parent_change_id": _metadata_id(metadata, "parent_change_id"),
        "parent_source_id": _metadata_id(metadata, "parent_source_id"),
    }
    state = (
        SourceLineageState.AVAILABLE
        if any(values.values())
        else SourceLineageState.MISSING_STRUCTURAL_CONTEXT
    )
    return StorySourceLineage(
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        content_hash=ref.content_hash,
        state=state,
        **values,
    )


def _anchors_from_lineage(lineage: StorySourceLineage) -> tuple[StoryEventAnchor, ...]:
    anchors: list[StoryEventAnchor] = []
    if lineage.explicit_change_id is not None:
        anchors.append(StoryEventAnchor(
            project_id=lineage.project_id,
            anchor_kind=StoryEventAnchorKind.EXPLICIT_CHANGE,
            strength=StoryEvidenceRelationStrength.STRONG,
            explicit_change_id=lineage.explicit_change_id,
        ))
    if lineage.parent_change_id is not None:
        anchors.append(StoryEventAnchor(
            project_id=lineage.project_id,
            anchor_kind=StoryEventAnchorKind.PARENT_CHANGE,
            strength=StoryEvidenceRelationStrength.MODERATE,
            parent_change_id=lineage.parent_change_id,
        ))
    if lineage.repository and lineage.commit_sha and lineage.symbol:
        anchors.append(StoryEventAnchor(
            project_id=lineage.project_id,
            anchor_kind=StoryEventAnchorKind.COMMIT_AND_SYMBOL,
            strength=StoryEvidenceRelationStrength.STRONG,
            repository=lineage.repository,
            commit_sha=lineage.commit_sha,
            symbol=lineage.symbol,
        ))
    if lineage.repository and lineage.commit_sha and lineage.file_path:
        anchors.append(StoryEventAnchor(
            project_id=lineage.project_id,
            anchor_kind=StoryEventAnchorKind.COMMIT_AND_PATH,
            strength=StoryEvidenceRelationStrength.MODERATE,
            repository=lineage.repository,
            commit_sha=lineage.commit_sha,
            file_path=lineage.file_path,
        ))
    anchors.append(StoryEventAnchor(
        project_id=lineage.project_id,
        anchor_kind=StoryEventAnchorKind.SOURCE_IDENTITY,
        strength=StoryEvidenceRelationStrength.WEAK,
        source_type=lineage.source_type,
        source_id=lineage.source_id,
    ))
    return _stable_anchors(anchors, project_id=lineage.project_id)


def _expected_project_boundary_subject_id(project_id: str) -> str:
    return (
        project_id
        if len(project_id) <= 100
        else build_project_evidence_stable_id(
            "pcb_", project_id, {"subject_type": "project"}
        )
    )


def _validate_boundaries(
    boundaries: Mapping[str, ProjectClaimBoundary],
    *,
    project_id: str,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
    capability_by_id: Mapping[str, ProjectCapabilityFact],
) -> None:
    for boundary in boundaries.values():
        if boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT:
            if boundary.subject_id not in evidence_by_id:
                _fail(
                    StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY,
                    boundary.boundary_id,
                )
        elif boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT:
            if boundary.subject_id not in capability_by_id:
                _fail(
                    StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY,
                    boundary.boundary_id,
                )
        elif boundary.subject_type is ClaimSubjectType.PROJECT:
            if boundary.subject_id != _expected_project_boundary_subject_id(project_id):
                _fail(
                    StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY,
                    boundary.boundary_id,
                )
        result = validate_project_claim_boundary(
            boundary,
            evidence_facts_by_id=evidence_by_id,
            capability_facts_by_id=capability_by_id,
        )
        if not result.valid:
            _fail(
                StoryEvidenceResolutionCode.INVALID_CLAIM_BOUNDARY,
                boundary.boundary_id,
            )


def _capability_lineage(
    capability: ProjectCapabilityFact,
    *,
    evidence_by_id: Mapping[str, ProjectEvidenceFact],
) -> CapabilityEvidenceLineage:
    evidence_ids = tuple(capability.source_evidence_fact_ids)
    for evidence_id in evidence_ids:
        fact = evidence_by_id.get(evidence_id)
        if fact is None or fact.project_id != capability.project_id:
            _fail(
                StoryEvidenceResolutionCode.INVALID_CAPABILITY_LINEAGE,
                capability.capability_id,
            )
    return CapabilityEvidenceLineage(
        project_id=capability.project_id,
        capability_id=capability.capability_id,
        capability_type=capability.capability_type,
        source_evidence_fact_ids=evidence_ids,
        present=capability.present,
        confidence=capability.confidence,
        metric_support=capability.metric_support,
        state=(
            CapabilityLineageState.RESOLVED
            if capability.present
            else CapabilityLineageState.NOT_PRESENT
        ),
    )


def _relevant_boundaries(
    boundaries: Mapping[str, ProjectClaimBoundary],
    *,
    project_id: str,
    evidence_ids: set[str],
    capability_ids: set[str],
) -> tuple[ProjectClaimBoundary, ...]:
    return tuple(sorted((
        boundary
        for boundary in boundaries.values()
        if (
            boundary.subject_type is ClaimSubjectType.PROJECT
            and boundary.project_id == project_id
        )
        or (
            boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
            and boundary.subject_id in evidence_ids
        )
        or (
            boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT
            and boundary.subject_id in capability_ids
        )
    ), key=lambda item: item.boundary_id))


def _relation(
    project_id: str,
    left: str,
    right: str,
    relation_type: StoryEvidenceRelationType,
    basis_ids: Sequence[str],
) -> StoryEvidenceRelation:
    first, second = sorted((left, right))
    return StoryEvidenceRelation(
        project_id=project_id,
        left_evidence_fact_id=first,
        right_evidence_fact_id=second,
        relation_type=relation_type,
        strength=_RELATION_STRENGTH[relation_type],
        basis_ids=tuple(basis_ids),
    )


def _build_relations(
    inputs: Sequence[StoryEvidenceInput],
) -> tuple[StoryEvidenceRelation, ...]:
    relations: set[StoryEvidenceRelation] = set()
    for left, right in combinations(inputs, 2):
        shared_capabilities = set(left.capability_ids) & set(right.capability_ids)
        for capability_id in shared_capabilities:
            relations.add(_relation(
                left.project_id,
                left.evidence_fact_id,
                right.evidence_fact_id,
                StoryEvidenceRelationType.CAPABILITY_SUPPORT_RELATION,
                (capability_id,),
            ))
        for left_lineage in left.source_lineages:
            for right_lineage in right.source_lineages:
                if (
                    left_lineage.explicit_change_id
                    and left_lineage.explicit_change_id
                    == right_lineage.explicit_change_id
                ):
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SAME_EXPLICIT_CHANGE,
                        (left_lineage.explicit_change_id,),
                    ))
                if (
                    left_lineage.parent_change_id
                    and left_lineage.parent_change_id == right_lineage.parent_change_id
                ):
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SAME_PARENT_CHANGE,
                        (left_lineage.parent_change_id,),
                    ))
                if (
                    left_lineage.repository
                    and left_lineage.repository == right_lineage.repository
                    and left_lineage.commit_sha
                    and left_lineage.commit_sha == right_lineage.commit_sha
                ):
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SAME_COMMIT,
                        (left_lineage.repository, left_lineage.commit_sha),
                    ))
                if left_lineage.identity_key == right_lineage.identity_key:
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SAME_SOURCE,
                        (left_lineage.source_type, left_lineage.source_id),
                    ))
                if (
                    left_lineage.repository
                    and left_lineage.repository == right_lineage.repository
                    and left_lineage.file_path
                    and left_lineage.file_path == right_lineage.file_path
                ):
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SAME_PATH,
                        (left_lineage.repository, left_lineage.file_path),
                    ))
                    if (
                        left_lineage.symbol
                        and left_lineage.symbol == right_lineage.symbol
                    ):
                        relations.add(_relation(
                            left.project_id,
                            left.evidence_fact_id,
                            right.evidence_fact_id,
                            StoryEvidenceRelationType.SAME_SYMBOL,
                            (
                                left_lineage.repository,
                                left_lineage.file_path,
                                left_lineage.symbol,
                            ),
                        ))
                if (
                    left_lineage.upstream_source_id
                    and left_lineage.upstream_source_id
                    == right_lineage.upstream_source_id
                ):
                    relations.add(_relation(
                        left.project_id,
                        left.evidence_fact_id,
                        right.evidence_fact_id,
                        StoryEvidenceRelationType.SHARED_VALIDATED_SOURCE_LINEAGE,
                        (left_lineage.upstream_source_id,),
                    ))
    if len(relations) > MAX_STORY_EVIDENCE_RELATIONS:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    return tuple(sorted(relations, key=_relation_sort_key))


def resolve_story_evidence_bundle(
    *,
    project_id: str,
    evidence_fact_ids: Sequence[str],
    evidence_facts: Sequence[ProjectEvidenceFact],
    capability_ids: Sequence[str] = (),
    capability_facts: Sequence[ProjectCapabilityFact] = (),
    claim_boundary_ids: Sequence[str] = (),
    claim_boundaries: Sequence[ProjectClaimBoundary] = (),
) -> StoryEvidenceBundle:
    """Resolve one bounded same-project structural bundle without I/O."""

    requested_project = _exact_project_id(project_id)
    evidence_by_id = _authority_index(
        evidence_facts,
        expected_type=ProjectEvidenceFact,
        id_attribute="evidence_fact_id",
        id_prefix="pef_",
        project_id=requested_project,
        conflict_code=StoryEvidenceResolutionCode.CONFLICTING_EVIDENCE_FACT,
    )
    capability_by_id = _authority_index(
        capability_facts,
        expected_type=ProjectCapabilityFact,
        id_attribute="capability_id",
        id_prefix="pcf_",
        project_id=requested_project,
        conflict_code=StoryEvidenceResolutionCode.CONFLICTING_CAPABILITY,
    )
    boundary_by_id = _authority_index(
        claim_boundaries,
        expected_type=ProjectClaimBoundary,
        id_attribute="boundary_id",
        id_prefix="pcb_",
        project_id=requested_project,
        conflict_code=StoryEvidenceResolutionCode.CONFLICTING_CLAIM_BOUNDARY,
    )
    for fact in evidence_by_id.values():
        if any(ref.project_id != requested_project for ref in fact.source_refs):
            _fail(
                StoryEvidenceResolutionCode.CROSS_PROJECT_AUTHORITY,
                fact.evidence_fact_id,
            )
    _validate_boundaries(
        boundary_by_id,
        project_id=requested_project,
        evidence_by_id=evidence_by_id,
        capability_by_id=capability_by_id,
    )
    selected_evidence_ids = set(_stable_authority_ids(
        evidence_fact_ids,
        "evidence_fact_ids",
        prefix="pef_",
        maximum=MAX_STORY_EVIDENCE_INPUTS,
    ))
    selected_capability_ids = _stable_authority_ids(
        capability_ids,
        "capability_ids",
        prefix="pcf_",
        maximum=MAX_STORY_CAPABILITY_LINEAGES,
    )
    requested_boundary_ids = _stable_authority_ids(
        claim_boundary_ids,
        "claim_boundary_ids",
        prefix="pcb_",
        maximum=MAX_STORY_CLAIM_BOUNDARIES,
    )
    for evidence_id in selected_evidence_ids:
        if evidence_id not in evidence_by_id:
            _fail(StoryEvidenceResolutionCode.UNKNOWN_EVIDENCE_FACT, evidence_id)
    capability_lineages: list[CapabilityEvidenceLineage] = []
    for capability_id in selected_capability_ids:
        capability = capability_by_id.get(capability_id)
        if capability is None:
            _fail(StoryEvidenceResolutionCode.UNKNOWN_CAPABILITY, capability_id)
        lineage = _capability_lineage(
            capability,
            evidence_by_id=evidence_by_id,
        )
        capability_lineages.append(lineage)
        if lineage.state is CapabilityLineageState.RESOLVED:
            selected_evidence_ids.update(lineage.source_evidence_fact_ids)
    if len(selected_evidence_ids) > MAX_STORY_EVIDENCE_INPUTS:
        _fail(StoryEvidenceResolutionCode.BOUND_EXCEEDED)
    if not selected_evidence_ids:
        _fail(StoryEvidenceResolutionCode.NO_EVIDENCE_INPUTS)
    relevant_boundaries = _relevant_boundaries(
        boundary_by_id,
        project_id=requested_project,
        evidence_ids=selected_evidence_ids,
        capability_ids=set(selected_capability_ids),
    )
    relevant_boundary_ids = {item.boundary_id for item in relevant_boundaries}
    for boundary_id in requested_boundary_ids:
        if boundary_id not in boundary_by_id:
            _fail(StoryEvidenceResolutionCode.UNKNOWN_CLAIM_BOUNDARY, boundary_id)
        if boundary_id not in relevant_boundary_ids:
            _fail(StoryEvidenceResolutionCode.IRRELEVANT_CLAIM_BOUNDARY, boundary_id)
    selected_boundary_ids = tuple(sorted(relevant_boundary_ids))
    source_lineage_authority: dict[tuple[str, str, str], StorySourceLineage] = {}
    inputs: list[StoryEvidenceInput] = []
    for evidence_id in sorted(selected_evidence_ids):
        fact = evidence_by_id[evidence_id]
        lineages = tuple(
            _lineage_from_source_ref(ref, project_id=requested_project)
            for ref in fact.source_refs
        )
        lineages = _stable_lineages(lineages, project_id=requested_project)
        for lineage in lineages:
            previous = source_lineage_authority.get(lineage.identity_key)
            if previous is not None and previous != lineage:
                _fail(
                    StoryEvidenceResolutionCode.CONFLICTING_SOURCE_LINEAGE,
                    lineage.source_id,
                )
            source_lineage_authority[lineage.identity_key] = lineage
        anchors = _stable_anchors(
            tuple(
                anchor
                for lineage in lineages
                for anchor in _anchors_from_lineage(lineage)
            ),
            project_id=requested_project,
        )
        attached_capability_ids = tuple(sorted(
            lineage.capability_id
            for lineage in capability_lineages
            if (
                lineage.state is CapabilityLineageState.RESOLVED
                and evidence_id in lineage.source_evidence_fact_ids
            )
        ))
        attached_boundary_ids = tuple(sorted(
            boundary.boundary_id
            for boundary in relevant_boundaries
            if boundary.subject_type is ClaimSubjectType.PROJECT
            or (
                boundary.subject_type is ClaimSubjectType.EVIDENCE_FACT
                and boundary.subject_id == evidence_id
            )
            or (
                boundary.subject_type is ClaimSubjectType.CAPABILITY_FACT
                and boundary.subject_id in attached_capability_ids
            )
        ))
        inputs.append(StoryEvidenceInput(
            project_id=requested_project,
            evidence_fact_id=evidence_id,
            evidence_type=fact.evidence_type,
            evidence_status=fact.status,
            confidence=fact.confidence,
            metric_support=fact.metric_support,
            technical_tags=tuple(fact.technical_tags),
            source_lineages=lineages,
            event_anchors=anchors,
            capability_ids=attached_capability_ids,
            claim_boundary_ids=attached_boundary_ids,
        ))
    ordered_inputs = tuple(sorted(inputs, key=_evidence_input_sort_key))
    bundle_anchors = _stable_anchors(
        tuple(anchor for item in ordered_inputs for anchor in item.event_anchors),
        project_id=requested_project,
    )
    lineage_state = (
        StoryEvidenceLineageState.MISSING_STRUCTURAL_CONTEXT
        if any(
            lineage.state is SourceLineageState.MISSING_STRUCTURAL_CONTEXT
            for item in ordered_inputs
            for lineage in item.source_lineages
        )
        else StoryEvidenceLineageState.COMPLETE
    )
    return StoryEvidenceBundle(
        project_id=requested_project,
        evidence_inputs=ordered_inputs,
        capability_lineages=tuple(capability_lineages),
        claim_boundary_ids=selected_boundary_ids,
        event_anchors=bundle_anchors,
        relations=_build_relations(ordered_inputs),
        lineage_state=lineage_state,
    )


def resolve_story_evidence_bundle_from_memory(
    *,
    project_memory: ProjectEvidenceMemory,
    evidence_fact_ids: Sequence[str],
    capability_ids: Sequence[str] = (),
    claim_boundary_ids: Sequence[str] = (),
) -> StoryEvidenceBundle:
    """Resolve from one already-loaded authoritative project memory object."""

    if not isinstance(project_memory, ProjectEvidenceMemory):
        raise TypeError("project_memory must be a ProjectEvidenceMemory")
    return resolve_story_evidence_bundle(
        project_id=project_memory.project_id,
        evidence_fact_ids=evidence_fact_ids,
        evidence_facts=tuple(project_memory.evidence_facts),
        capability_ids=capability_ids,
        capability_facts=tuple(project_memory.capability_facts),
        claim_boundary_ids=claim_boundary_ids,
        claim_boundaries=tuple(project_memory.claim_boundaries),
    )


__all__ = [
    "CapabilityEvidenceLineage",
    "CapabilityLineageState",
    "MAX_EVENT_ANCHORS",
    "MAX_SOURCE_LINEAGES_PER_INPUT",
    "MAX_STORY_CAPABILITY_LINEAGES",
    "MAX_STORY_CLAIM_BOUNDARIES",
    "MAX_STORY_EVIDENCE_INPUTS",
    "MAX_STORY_EVIDENCE_RELATIONS",
    "SourceLineageState",
    "StoryEventAnchor",
    "StoryEventAnchorKind",
    "StoryEvidenceBundle",
    "StoryEvidenceInput",
    "StoryEvidenceLineageState",
    "StoryEvidenceRelation",
    "StoryEvidenceRelationStrength",
    "StoryEvidenceRelationType",
    "StoryEvidenceResolutionCode",
    "StoryEvidenceResolutionError",
    "StorySourceLineage",
    "resolve_story_evidence_bundle",
    "resolve_story_evidence_bundle_from_memory",
]
