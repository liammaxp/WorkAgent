"""Deterministic revision and lifecycle management for canonical stories.

This boundary consumes immutable Step-7 candidate snapshots and the accepted
Step-8 canonical identity result.  It compares already-recomputed semantic
state; it does not retrieve evidence, reconstruct stories, or rematch identity.

The lifecycle memory defined here is deliberately an in-memory semantic
contract.  ``engineering_story_memory.v1`` remains the candidate-snapshot
artifact until a later materialization step defines a production envelope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any

from backend.engineering_story_matching import (
    CanonicalEngineeringStoryIdentity,
    EngineeringStoryIdentityMap,
    StoryCanonicalizationResult,
    StoryMergeOutcome,
    StorySplitOutcome,
    resolve_canonical_story_alias,
)
from backend.engineering_story_memory import EngineeringStoryMemoryRecord
from backend.engineering_story_models import (
    EngineeringStoryContract,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    StoryFieldEvidenceState,
    validate_engineering_story_id,
)
from backend.engineering_story_reconstruction import (
    StoryReconstructionDiagnosticCode,
)
from backend.engineering_story_sufficiency import SufficiencyDiagnosticCode
from backend.project_repository_identity import normalize_project_id


MAX_ENGINEERING_STORY_REVISION_HISTORIES = 2_048
MAX_ENGINEERING_STORY_REVISIONS_PER_CANONICAL = 256
MAX_ENGINEERING_STORY_LIFECYCLE_RESULTS = 2_048
MAX_ENGINEERING_STORY_LIFECYCLE_DIAGNOSTICS = 32
MAX_REVISION_PROVENANCE_DIFF_IDS = 256

_CANONICAL_STORY_ID_RE = re.compile(r"^engineering_story_[0-9a-f]{24}$")
_REVISION_ID_RE = re.compile(r"^engineering_story_revision_[0-9a-f]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_FIELD_INDEX = {value: index for index, value in enumerate(_FIELD_ORDER)}


class StoryRevisionChangeType(str, Enum):
    INITIAL_REVISION = "initial_revision"
    RECORD_CONTENT_CHANGED = "record_content_changed"
    PROVENANCE_ADDED = "provenance_added"
    PROVENANCE_REMOVED = "provenance_removed"
    PROVENANCE_REPLACED = "provenance_replaced"
    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_CHANGED = "field_changed"
    FIELD_SUPPORT_UPGRADED = "field_support_upgraded"
    FIELD_SUPPORT_DOWNGRADED = "field_support_downgraded"
    CLAIM_BOUNDARY_CHANGED = "claim_boundary_changed"
    CLAIM_SUFFICIENCY_CHANGED = "claim_sufficiency_changed"
    STORY_SUFFICIENCY_CHANGED = "story_sufficiency_changed"
    OPPORTUNITY_CHANGED = "opportunity_changed"
    CONFLICT_INTRODUCED = "conflict_introduced"
    CONFLICT_RESOLVED = "conflict_resolved"
    REVALIDATION_REQUIRED = "revalidation_required"
    REVALIDATION_SUCCEEDED = "revalidation_succeeded"
    REVALIDATION_FAILED = "revalidation_failed"
    CANONICAL_MERGE_RELATION = "canonical_merge_relation"
    CANONICAL_SPLIT_RELATION = "canonical_split_relation"


class StoryLifecycleReasonCode(str, Enum):
    INITIALIZED_ACTIVE = "initialized_active"
    INITIALIZED_CONFLICTED = "initialized_conflicted"
    SEMANTIC_STATE_UPDATED = "semantic_state_updated"
    FIELD_SUPPORT_REQUIRES_REVALIDATION = (
        "field_support_requires_revalidation"
    )
    CORE_SUPPORT_BECAME_STALE = "core_support_became_stale"
    AUTHORITATIVE_CONFLICT = "authoritative_conflict"
    EXPLICIT_REVALIDATION_SUCCEEDED = "explicit_revalidation_succeeded"
    EXPLICIT_REVALIDATION_FAILED = "explicit_revalidation_failed"
    CANONICAL_MERGE_SUPERSEDED = "canonical_merge_superseded"
    CANONICAL_SPLIT_PRESERVED = "canonical_split_preserved"
    OPPORTUNITY_ONLY_UPDATE = "opportunity_only_update"


class StoryRevalidationOutcome(str, Enum):
    UNCHANGED = "unchanged"
    VALIDATED_ACTIVE = "validated_active"
    UPDATED_ACTIVE = "updated_active"
    REQUIRES_REVALIDATION = "requires_revalidation"
    STALE = "stale"
    CONFLICTED = "conflicted"
    SUPERSEDED = "superseded"


class StoryLifecycleDiagnosticCode(str, Enum):
    AMBIGUOUS_IDENTITY_PRESERVED = "ambiguous_identity_preserved"
    AMBIGUOUS_SPLIT_PRESERVED = "ambiguous_split_preserved"
    MISSING_AUTHORITATIVE_CANDIDATE_MARKED_STALE = (
        "missing_authoritative_candidate_marked_stale"
    )


class EngineeringStoryLifecycleErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    CROSS_PROJECT_INPUT = "cross_project_input"
    UNRESOLVED_CANONICAL_IDENTITY = "unresolved_canonical_identity"
    CONFLICTING_CANONICAL_HISTORY = "conflicting_canonical_history"
    CONFLICTING_REVISION_ID = "conflicting_revision_id"
    MISSING_PARENT_REVISION = "missing_parent_revision"
    REVISION_CYCLE = "revision_cycle"
    BRANCHED_REVISION_HISTORY = "branched_revision_history"
    INVALID_CURRENT_REVISION = "invalid_current_revision"
    INVALID_LIFECYCLE_TRANSITION = "invalid_lifecycle_transition"
    INVALID_CHANGE_CLASSIFICATION = "invalid_change_classification"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    IDENTITY_HISTORY_MISMATCH = "identity_history_mismatch"
    UNSAFE_CALLER_LIFECYCLE = "unsafe_caller_lifecycle"


class EngineeringStoryLifecycleError(ValueError):
    """Bounded fail-closed lifecycle-contract error."""

    def __init__(self, code: EngineeringStoryLifecycleErrorCode | str) -> None:
        self.code = EngineeringStoryLifecycleErrorCode(code)
        super().__init__(self.code.value)


def _fail(code: EngineeringStoryLifecycleErrorCode) -> None:
    raise EngineeringStoryLifecycleError(code)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    return normalized


def _canonical_story_id(value: Any) -> str:
    story_id = validate_engineering_story_id(value)
    if not _CANONICAL_STORY_ID_RE.fullmatch(story_id):
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    return story_id


def _revision_id(value: Any) -> str:
    if not isinstance(value, str) or not _REVISION_ID_RE.fullmatch(value):
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    return value


def _stable_enums(
    values: Sequence[Any],
    enum_type: type[Enum],
    *,
    maximum: int,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
    index = {value: position for position, value in enumerate(enum_type)}
    normalized = {enum_type(value) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__))


def _authority_id(value: Any, kind: str) -> str:
    kwargs = {
        "evidence_fact_ids": (),
        "capability_fact_ids": (),
        "claim_boundary_ids": (),
    }
    kwargs[kind] = (value,)
    try:
        validated = EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            **kwargs,
        )
    except (TypeError, ValueError) as exc:
        raise EngineeringStoryLifecycleError(
            EngineeringStoryLifecycleErrorCode.INVALID_INPUT
        ) from exc
    return getattr(validated, kind)[0]


def _stable_authority_ids(
    values: Sequence[str],
    *,
    kind: str,
    maximum: int = MAX_REVISION_PROVENANCE_DIFF_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{kind} must be a sequence")
    if len(values) > maximum:
        _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
    return tuple(sorted({_authority_id(value, kind) for value in values}))


def _stable_digests(
    values: Sequence[str],
    *,
    maximum: int = MAX_REVISION_PROVENANCE_DIFF_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("fingerprints must be a sequence")
    if len(values) > maximum:
        _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
    return tuple(sorted({_digest(value) for value in values}))


def _stable_canonical_ids(
    values: Sequence[str],
    *,
    maximum: int = MAX_ENGINEERING_STORY_LIFECYCLE_RESULTS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("canonical story IDs must be a sequence")
    if len(values) > maximum:
        _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
    return tuple(sorted({_canonical_story_id(value) for value in values}))


@dataclass(frozen=True, slots=True)
class StoryRevisionProvenanceDelta(EngineeringStoryContract):
    added_evidence_fact_ids: tuple[str, ...] = ()
    removed_evidence_fact_ids: tuple[str, ...] = ()
    added_capability_fact_ids: tuple[str, ...] = ()
    removed_capability_fact_ids: tuple[str, ...] = ()
    added_claim_boundary_ids: tuple[str, ...] = ()
    removed_claim_boundary_ids: tuple[str, ...] = ()
    added_source_lineage_fingerprints: tuple[str, ...] = ()
    removed_source_lineage_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        specs = (
            ("added_evidence_fact_ids", "evidence_fact_ids"),
            ("removed_evidence_fact_ids", "evidence_fact_ids"),
            ("added_capability_fact_ids", "capability_fact_ids"),
            ("removed_capability_fact_ids", "capability_fact_ids"),
            ("added_claim_boundary_ids", "claim_boundary_ids"),
            ("removed_claim_boundary_ids", "claim_boundary_ids"),
        )
        for attribute, kind in specs:
            object.__setattr__(
                self,
                attribute,
                _stable_authority_ids(getattr(self, attribute), kind=kind),
            )
        object.__setattr__(
            self,
            "added_source_lineage_fingerprints",
            _stable_digests(self.added_source_lineage_fingerprints),
        )
        object.__setattr__(
            self,
            "removed_source_lineage_fingerprints",
            _stable_digests(self.removed_source_lineage_fingerprints),
        )
        pairs = (
            (self.added_evidence_fact_ids, self.removed_evidence_fact_ids),
            (self.added_capability_fact_ids, self.removed_capability_fact_ids),
            (self.added_claim_boundary_ids, self.removed_claim_boundary_ids),
            (
                self.added_source_lineage_fingerprints,
                self.removed_source_lineage_fingerprints,
            ),
        )
        if any(set(added) & set(removed) for added, removed in pairs):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)

    @property
    def has_added(self) -> bool:
        return any((
            self.added_evidence_fact_ids,
            self.added_capability_fact_ids,
            self.added_claim_boundary_ids,
            self.added_source_lineage_fingerprints,
        ))

    @property
    def has_removed(self) -> bool:
        return any((
            self.removed_evidence_fact_ids,
            self.removed_capability_fact_ids,
            self.removed_claim_boundary_ids,
            self.removed_source_lineage_fingerprints,
        ))


_FIELD_CHANGE_TYPES = {
    StoryRevisionChangeType.FIELD_ADDED,
    StoryRevisionChangeType.FIELD_REMOVED,
    StoryRevisionChangeType.FIELD_CHANGED,
    StoryRevisionChangeType.FIELD_SUPPORT_UPGRADED,
    StoryRevisionChangeType.FIELD_SUPPORT_DOWNGRADED,
    StoryRevisionChangeType.CLAIM_BOUNDARY_CHANGED,
}


@dataclass(frozen=True, slots=True)
class StoryFieldSupportChange(EngineeringStoryContract):
    field_name: EngineeringStoryFieldName
    previous_state: StoryFieldEvidenceState
    current_state: StoryFieldEvidenceState
    change_types: tuple[StoryRevisionChangeType, ...]
    added_evidence_fact_ids: tuple[str, ...] = ()
    removed_evidence_fact_ids: tuple[str, ...] = ()
    added_capability_fact_ids: tuple[str, ...] = ()
    removed_capability_fact_ids: tuple[str, ...] = ()
    added_claim_boundary_ids: tuple[str, ...] = ()
    removed_claim_boundary_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        field_name = EngineeringStoryFieldName(self.field_name)
        previous = StoryFieldEvidenceState(self.previous_state)
        current = StoryFieldEvidenceState(self.current_state)
        changes = _stable_enums(
            self.change_types,
            StoryRevisionChangeType,
            maximum=len(_FIELD_CHANGE_TYPES),
            name="change_types",
        )
        if not changes or any(change not in _FIELD_CHANGE_TYPES for change in changes):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
        specs = (
            ("added_evidence_fact_ids", "evidence_fact_ids"),
            ("removed_evidence_fact_ids", "evidence_fact_ids"),
            ("added_capability_fact_ids", "capability_fact_ids"),
            ("removed_capability_fact_ids", "capability_fact_ids"),
            ("added_claim_boundary_ids", "claim_boundary_ids"),
            ("removed_claim_boundary_ids", "claim_boundary_ids"),
        )
        for attribute, kind in specs:
            object.__setattr__(
                self,
                attribute,
                _stable_authority_ids(
                    getattr(self, attribute),
                    kind=kind,
                    maximum=64,
                ),
            )
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "previous_state", previous)
        object.__setattr__(self, "current_state", current)
        object.__setattr__(self, "change_types", changes)


_ALLOWED_STATUS_TRANSITIONS = {
    EngineeringStoryStatus.ACTIVE: {
        EngineeringStoryStatus.ACTIVE,
        EngineeringStoryStatus.STALE,
        EngineeringStoryStatus.CONFLICTED,
        EngineeringStoryStatus.SUPERSEDED,
    },
    EngineeringStoryStatus.STALE: {
        EngineeringStoryStatus.STALE,
        EngineeringStoryStatus.ACTIVE,
        EngineeringStoryStatus.CONFLICTED,
        EngineeringStoryStatus.SUPERSEDED,
    },
    EngineeringStoryStatus.CONFLICTED: {
        EngineeringStoryStatus.CONFLICTED,
        EngineeringStoryStatus.ACTIVE,
        EngineeringStoryStatus.SUPERSEDED,
    },
    EngineeringStoryStatus.SUPERSEDED: {
        EngineeringStoryStatus.SUPERSEDED,
    },
}


@dataclass(frozen=True, slots=True)
class StoryLifecycleDecision(EngineeringStoryContract):
    project_id: str
    canonical_story_id: str
    previous_lifecycle: EngineeringStoryLifecycle | None
    resulting_lifecycle: EngineeringStoryLifecycle
    reason_codes: tuple[StoryLifecycleReasonCode, ...]
    affected_fields: tuple[EngineeringStoryFieldName, ...] = ()

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        canonical_id = _canonical_story_id(self.canonical_story_id)
        if self.previous_lifecycle is not None and not isinstance(
            self.previous_lifecycle, EngineeringStoryLifecycle
        ):
            raise TypeError("previous_lifecycle must be EngineeringStoryLifecycle")
        if not isinstance(self.resulting_lifecycle, EngineeringStoryLifecycle):
            raise TypeError("resulting_lifecycle must be EngineeringStoryLifecycle")
        reasons = _stable_enums(
            self.reason_codes,
            StoryLifecycleReasonCode,
            maximum=len(StoryLifecycleReasonCode),
            name="reason_codes",
        )
        if not reasons:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        fields = _stable_enums(
            self.affected_fields,
            EngineeringStoryFieldName,
            maximum=len(_FIELD_ORDER),
            name="affected_fields",
        )
        previous = self.previous_lifecycle
        current = self.resulting_lifecycle
        if previous is None:
            if current.status not in {
                EngineeringStoryStatus.ACTIVE,
                EngineeringStoryStatus.CONFLICTED,
            }:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                )
            required_initial_reason = (
                StoryLifecycleReasonCode.INITIALIZED_CONFLICTED
                if current.status is EngineeringStoryStatus.CONFLICTED
                else StoryLifecycleReasonCode.INITIALIZED_ACTIVE
            )
            if required_initial_reason not in reasons:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                )
        elif current.status not in _ALLOWED_STATUS_TRANSITIONS[previous.status]:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        if (
            previous is not None
            and previous.status in {
                EngineeringStoryStatus.STALE,
                EngineeringStoryStatus.CONFLICTED,
            }
            and current.status is EngineeringStoryStatus.ACTIVE
            and StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_SUCCEEDED
            not in reasons
        ):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        if (
            previous is not None
            and previous.requires_revalidation
            and current.status is EngineeringStoryStatus.ACTIVE
            and not current.requires_revalidation
            and StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_SUCCEEDED
            not in reasons
        ):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        if (
            current.status is EngineeringStoryStatus.CONFLICTED
            and StoryLifecycleReasonCode.AUTHORITATIVE_CONFLICT not in reasons
            and StoryLifecycleReasonCode.INITIALIZED_CONFLICTED not in reasons
            and (
                previous is None
                or previous.status is not EngineeringStoryStatus.CONFLICTED
            )
        ):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        if current.status is EngineeringStoryStatus.SUPERSEDED:
            if StoryLifecycleReasonCode.CANONICAL_MERGE_SUPERSEDED not in reasons:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                )
        elif StoryLifecycleReasonCode.CANONICAL_MERGE_SUPERSEDED in reasons:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "affected_fields", fields)


def _revision_payload(
    *,
    canonical_story_id: str,
    project_id: str,
    record: EngineeringStoryMemoryRecord,
    parent_revision_id: str | None,
    change_types: Sequence[StoryRevisionChangeType],
    provenance_delta: StoryRevisionProvenanceDelta,
    field_changes: Sequence[StoryFieldSupportChange],
    lifecycle_decision: StoryLifecycleDecision,
    revalidation_outcome: StoryRevalidationOutcome,
) -> dict[str, Any]:
    return {
        "canonical_story_id": canonical_story_id,
        "project_id": project_id,
        "candidate_story_id": record.candidate_story_id,
        "cluster_id": record.cluster_id,
        "record_fingerprint": record.record_fingerprint,
        "provenance_fingerprint": record.provenance.provenance_fingerprint,
        "parent_revision_id": parent_revision_id,
        "change_types": [item.value for item in change_types],
        "provenance_delta": provenance_delta.to_dict(),
        "field_changes": [item.to_dict() for item in field_changes],
        "lifecycle_decision": lifecycle_decision.to_dict(),
        "revalidation_outcome": revalidation_outcome.value,
    }


def build_engineering_story_revision_id(
    *,
    canonical_story_id: str,
    project_id: str,
    record: EngineeringStoryMemoryRecord,
    parent_revision_id: str | None,
    change_types: Sequence[StoryRevisionChangeType],
    provenance_delta: StoryRevisionProvenanceDelta,
    field_changes: Sequence[StoryFieldSupportChange],
    lifecycle_decision: StoryLifecycleDecision,
    revalidation_outcome: StoryRevalidationOutcome,
) -> str:
    canonical_id = _canonical_story_id(canonical_story_id)
    project = _exact_project_id(project_id)
    if not isinstance(record, EngineeringStoryMemoryRecord):
        raise TypeError("record must be EngineeringStoryMemoryRecord")
    parent = None if parent_revision_id is None else _revision_id(parent_revision_id)
    changes = _stable_enums(
        change_types,
        StoryRevisionChangeType,
        maximum=len(StoryRevisionChangeType),
        name="change_types",
    )
    if not isinstance(provenance_delta, StoryRevisionProvenanceDelta):
        raise TypeError("provenance_delta must be StoryRevisionProvenanceDelta")
    if isinstance(field_changes, (str, bytes)) or not isinstance(
        field_changes, Sequence
    ):
        raise TypeError("field_changes must be a sequence")
    fields = tuple(sorted(set(field_changes), key=lambda item: _FIELD_INDEX[item.field_name]))
    if not isinstance(lifecycle_decision, StoryLifecycleDecision):
        raise TypeError("lifecycle_decision must be StoryLifecycleDecision")
    outcome = StoryRevalidationOutcome(revalidation_outcome)
    digest = _fingerprint(_revision_payload(
        canonical_story_id=canonical_id,
        project_id=project,
        record=record,
        parent_revision_id=parent,
        change_types=changes,
        provenance_delta=provenance_delta,
        field_changes=fields,
        lifecycle_decision=lifecycle_decision,
        revalidation_outcome=outcome,
    ))[:24]
    return f"engineering_story_revision_{digest}"


@dataclass(frozen=True, slots=True)
class EngineeringStoryRevision(EngineeringStoryContract):
    canonical_story_id: str
    revision_id: str
    project_id: str
    record: EngineeringStoryMemoryRecord
    provenance_fingerprint: str
    record_fingerprint: str
    parent_revision_id: str | None
    change_types: tuple[StoryRevisionChangeType, ...]
    provenance_delta: StoryRevisionProvenanceDelta
    field_changes: tuple[StoryFieldSupportChange, ...]
    lifecycle_decision: StoryLifecycleDecision
    revalidation_outcome: StoryRevalidationOutcome

    def __post_init__(self) -> None:
        canonical_id = _canonical_story_id(self.canonical_story_id)
        project_id = _exact_project_id(self.project_id)
        if not isinstance(self.record, EngineeringStoryMemoryRecord):
            raise TypeError("record must be EngineeringStoryMemoryRecord")
        if self.record.project_id != project_id:
            _fail(EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT)
        provenance_fingerprint = _digest(self.provenance_fingerprint)
        record_fingerprint = _digest(self.record_fingerprint)
        if (
            provenance_fingerprint
            != self.record.provenance.provenance_fingerprint
            or record_fingerprint != self.record.record_fingerprint
        ):
            _fail(EngineeringStoryLifecycleErrorCode.FINGERPRINT_MISMATCH)
        parent = (
            None
            if self.parent_revision_id is None
            else _revision_id(self.parent_revision_id)
        )
        changes = _stable_enums(
            self.change_types,
            StoryRevisionChangeType,
            maximum=len(StoryRevisionChangeType),
            name="change_types",
        )
        if not changes:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
        if not isinstance(self.provenance_delta, StoryRevisionProvenanceDelta):
            raise TypeError("provenance_delta must be StoryRevisionProvenanceDelta")
        if (
            isinstance(self.field_changes, (str, bytes))
            or not isinstance(self.field_changes, Sequence)
            or any(
                not isinstance(item, StoryFieldSupportChange)
                for item in self.field_changes
            )
        ):
            raise TypeError("field_changes must contain StoryFieldSupportChange")
        if len(self.field_changes) > len(_FIELD_ORDER):
            _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
        fields_by_name: dict[EngineeringStoryFieldName, StoryFieldSupportChange] = {}
        for item in self.field_changes:
            previous = fields_by_name.get(item.field_name)
            if previous is not None and previous != item:
                _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
            fields_by_name[item.field_name] = item
        field_changes = tuple(
            fields_by_name[name] for name in _FIELD_ORDER if name in fields_by_name
        )
        if not isinstance(self.lifecycle_decision, StoryLifecycleDecision):
            raise TypeError("lifecycle_decision must be StoryLifecycleDecision")
        if (
            self.lifecycle_decision.project_id != project_id
            or self.lifecycle_decision.canonical_story_id != canonical_id
        ):
            _fail(EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT)
        outcome = StoryRevalidationOutcome(self.revalidation_outcome)
        if parent is None:
            if StoryRevisionChangeType.INITIAL_REVISION not in changes:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION
                )
            if self.lifecycle_decision.previous_lifecycle is not None:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                )
        elif StoryRevisionChangeType.INITIAL_REVISION in changes:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION)
        elif self.lifecycle_decision.previous_lifecycle is None:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        expected_outcomes = {
            EngineeringStoryStatus.ACTIVE: {
                StoryRevalidationOutcome.VALIDATED_ACTIVE,
                StoryRevalidationOutcome.UPDATED_ACTIVE,
                StoryRevalidationOutcome.REQUIRES_REVALIDATION,
            },
            EngineeringStoryStatus.STALE: {StoryRevalidationOutcome.STALE},
            EngineeringStoryStatus.CONFLICTED: {
                StoryRevalidationOutcome.CONFLICTED
            },
            EngineeringStoryStatus.SUPERSEDED: {
                StoryRevalidationOutcome.SUPERSEDED
            },
        }
        if outcome not in expected_outcomes[self.lifecycle_decision.resulting_lifecycle.status]:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
        if (
            outcome is StoryRevalidationOutcome.REQUIRES_REVALIDATION
        ) != self.lifecycle_decision.resulting_lifecycle.requires_revalidation:
            if self.lifecycle_decision.resulting_lifecycle.status is EngineeringStoryStatus.ACTIVE:
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                )
        if (
            self.provenance_delta.has_added
            != (StoryRevisionChangeType.PROVENANCE_ADDED in changes)
            or self.provenance_delta.has_removed
            != (StoryRevisionChangeType.PROVENANCE_REMOVED in changes)
            or (
                self.provenance_delta.has_added
                and self.provenance_delta.has_removed
            )
            != (StoryRevisionChangeType.PROVENANCE_REPLACED in changes)
        ):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION)
        if any(
            not set(item.change_types).issubset(changes)
            for item in field_changes
        ):
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION)
        expected = build_engineering_story_revision_id(
            canonical_story_id=canonical_id,
            project_id=project_id,
            record=self.record,
            parent_revision_id=parent,
            change_types=changes,
            provenance_delta=self.provenance_delta,
            field_changes=field_changes,
            lifecycle_decision=self.lifecycle_decision,
            revalidation_outcome=outcome,
        )
        if _revision_id(self.revision_id) != expected:
            _fail(EngineeringStoryLifecycleErrorCode.FINGERPRINT_MISMATCH)
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "provenance_fingerprint", provenance_fingerprint)
        object.__setattr__(self, "record_fingerprint", record_fingerprint)
        object.__setattr__(self, "parent_revision_id", parent)
        object.__setattr__(self, "change_types", changes)
        object.__setattr__(self, "field_changes", field_changes)
        object.__setattr__(self, "revalidation_outcome", outcome)

    @property
    def lifecycle(self) -> EngineeringStoryLifecycle:
        return self.lifecycle_decision.resulting_lifecycle

    @property
    def candidate_story_id(self) -> str:
        return self.record.candidate_story_id

    @property
    def cluster_id(self) -> str:
        return self.record.cluster_id


@dataclass(frozen=True, slots=True)
class EngineeringStoryRevisionHistory(EngineeringStoryContract):
    founding_identity: CanonicalEngineeringStoryIdentity
    revisions: tuple[EngineeringStoryRevision, ...]
    current_revision_id: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.founding_identity, CanonicalEngineeringStoryIdentity
        ):
            raise TypeError("founding_identity must be canonical identity")
        if isinstance(self.revisions, (str, bytes)) or not isinstance(
            self.revisions, Sequence
        ):
            raise TypeError("revisions must be a sequence")
        if (
            not self.revisions
            or len(self.revisions) > MAX_ENGINEERING_STORY_REVISIONS_PER_CANONICAL
        ):
            _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
        by_id: dict[str, EngineeringStoryRevision] = {}
        by_payload: dict[str, str] = {}
        for revision in self.revisions:
            if not isinstance(revision, EngineeringStoryRevision):
                raise TypeError("revisions must contain EngineeringStoryRevision")
            if (
                revision.canonical_story_id
                != self.founding_identity.canonical_story_id
                or revision.project_id != self.founding_identity.project_id
            ):
                _fail(EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT)
            previous = by_id.get(revision.revision_id)
            if previous is not None and previous != revision:
                _fail(EngineeringStoryLifecycleErrorCode.CONFLICTING_REVISION_ID)
            by_id[revision.revision_id] = revision
            semantic = _canonical_json({
                key: value
                for key, value in revision.to_dict().items()
                if key != "revision_id"
            })
            prior_id = by_payload.get(semantic)
            if prior_id is not None and prior_id != revision.revision_id:
                _fail(EngineeringStoryLifecycleErrorCode.CONFLICTING_REVISION_ID)
            by_payload[semantic] = revision.revision_id
        current_id = _revision_id(self.current_revision_id)
        if current_id not in by_id:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_CURRENT_REVISION)
        roots = [item for item in by_id.values() if item.parent_revision_id is None]
        if len(roots) != 1:
            _fail(EngineeringStoryLifecycleErrorCode.MISSING_PARENT_REVISION)
        children: dict[str, list[str]] = {}
        for revision in by_id.values():
            parent_id = revision.parent_revision_id
            if parent_id is None:
                if revision.lifecycle_decision.previous_lifecycle is not None:
                    _fail(
                        EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION
                    )
                continue
            parent = by_id.get(parent_id)
            if parent is None:
                _fail(EngineeringStoryLifecycleErrorCode.MISSING_PARENT_REVISION)
            if revision.lifecycle_decision.previous_lifecycle != parent.lifecycle:
                _fail(EngineeringStoryLifecycleErrorCode.INVALID_LIFECYCLE_TRANSITION)
            expected_changes, expected_delta, expected_fields = _change_summary(
                parent.record,
                revision.record,
            )
            allowed_lifecycle_changes = {
                StoryRevisionChangeType.CONFLICT_INTRODUCED,
                StoryRevisionChangeType.CONFLICT_RESOLVED,
                StoryRevisionChangeType.REVALIDATION_REQUIRED,
                StoryRevisionChangeType.REVALIDATION_SUCCEEDED,
                StoryRevisionChangeType.REVALIDATION_FAILED,
                StoryRevisionChangeType.CANONICAL_MERGE_RELATION,
                StoryRevisionChangeType.CANONICAL_SPLIT_RELATION,
            }
            if (
                revision.provenance_delta != expected_delta
                or revision.field_changes != expected_fields
                or not expected_changes.issubset(set(revision.change_types))
                or not (
                    set(revision.change_types) - expected_changes
                ).issubset(allowed_lifecycle_changes)
            ):
                _fail(
                    EngineeringStoryLifecycleErrorCode.INVALID_CHANGE_CLASSIFICATION
                )
            children.setdefault(parent_id, []).append(revision.revision_id)
        if any(len(values) > 1 for values in children.values()):
            _fail(EngineeringStoryLifecycleErrorCode.BRANCHED_REVISION_HISTORY)
        ordered: list[EngineeringStoryRevision] = []
        visited: set[str] = set()
        cursor: EngineeringStoryRevision | None = roots[0]
        while cursor is not None:
            if cursor.revision_id in visited:
                _fail(EngineeringStoryLifecycleErrorCode.REVISION_CYCLE)
            visited.add(cursor.revision_id)
            ordered.append(cursor)
            next_ids = children.get(cursor.revision_id, ())
            cursor = by_id[next_ids[0]] if next_ids else None
        if len(visited) != len(by_id):
            _fail(EngineeringStoryLifecycleErrorCode.REVISION_CYCLE)
        if ordered[-1].revision_id != current_id:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_CURRENT_REVISION)
        object.__setattr__(self, "revisions", tuple(ordered))
        object.__setattr__(self, "current_revision_id", current_id)

    @property
    def project_id(self) -> str:
        return self.founding_identity.project_id

    @property
    def canonical_story_id(self) -> str:
        return self.founding_identity.canonical_story_id

    @property
    def current_revision(self) -> EngineeringStoryRevision:
        return self.revisions[-1]

    @property
    def current_revision_number(self) -> int:
        return len(self.revisions)


def _lifecycle_memory_payload(
    memory: "EngineeringStoryLifecycleMemory",
) -> dict[str, Any]:
    return {
        "identity_map": memory.identity_map.to_dict(),
        "histories": [item.to_dict() for item in memory.histories],
    }


@dataclass(frozen=True, slots=True)
class EngineeringStoryLifecycleMemory(EngineeringStoryContract):
    identity_map: EngineeringStoryIdentityMap
    histories: tuple[EngineeringStoryRevisionHistory, ...] = ()
    logical_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.identity_map, EngineeringStoryIdentityMap):
            raise TypeError("identity_map must be EngineeringStoryIdentityMap")
        if isinstance(self.histories, (str, bytes)) or not isinstance(
            self.histories, Sequence
        ):
            raise TypeError("histories must be a sequence")
        if len(self.histories) > MAX_ENGINEERING_STORY_REVISION_HISTORIES:
            _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
        identities = {
            item.canonical_story_id: item
            for item in self.identity_map.canonical_identities
        }
        by_id: dict[str, EngineeringStoryRevisionHistory] = {}
        for history in self.histories:
            if not isinstance(history, EngineeringStoryRevisionHistory):
                raise TypeError("histories must contain revision histories")
            identity = identities.get(history.canonical_story_id)
            if identity is None or identity != history.founding_identity:
                _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
            previous = by_id.get(history.canonical_story_id)
            if previous is not None and previous != history:
                _fail(
                    EngineeringStoryLifecycleErrorCode.CONFLICTING_CANONICAL_HISTORY
                )
            by_id[history.canonical_story_id] = history
        aliases = {
            item.alias_story_id: item.canonical_story_id
            for item in self.identity_map.aliases
        }
        for alias_id, target_id in aliases.items():
            history = by_id.get(alias_id)
            if history is None:
                continue
            lifecycle = history.current_revision.lifecycle
            if (
                lifecycle.status is not EngineeringStoryStatus.SUPERSEDED
                or lifecycle.superseded_by_story_id != target_id
            ):
                _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
        histories = tuple(sorted(
            by_id.values(),
            key=lambda item: (
                item.project_id.casefold(),
                item.project_id,
                item.canonical_story_id,
            ),
        ))
        object.__setattr__(self, "histories", histories)
        expected = _fingerprint(_lifecycle_memory_payload(self))
        if self.logical_fingerprint not in ("", expected):
            _fail(EngineeringStoryLifecycleErrorCode.FINGERPRINT_MISMATCH)
        object.__setattr__(self, "logical_fingerprint", expected)

    def history_for(
        self, canonical_story_id: str
    ) -> EngineeringStoryRevisionHistory | None:
        story_id = _canonical_story_id(canonical_story_id)
        return next(
            (
                item
                for item in self.histories
                if item.canonical_story_id == story_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class StoryRevalidationResult(EngineeringStoryContract):
    project_id: str
    canonical_story_id: str
    previous_revision_id: str | None
    current_revision_id: str
    outcome: StoryRevalidationOutcome
    revision_created: bool
    lifecycle_decision: StoryLifecycleDecision

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        canonical_id = _canonical_story_id(self.canonical_story_id)
        previous = (
            None
            if self.previous_revision_id is None
            else _revision_id(self.previous_revision_id)
        )
        current = _revision_id(self.current_revision_id)
        outcome = StoryRevalidationOutcome(self.outcome)
        if not isinstance(self.revision_created, bool):
            raise TypeError("revision_created must be a boolean")
        if not isinstance(self.lifecycle_decision, StoryLifecycleDecision):
            raise TypeError("lifecycle_decision must be StoryLifecycleDecision")
        if (
            self.lifecycle_decision.project_id != project_id
            or self.lifecycle_decision.canonical_story_id != canonical_id
        ):
            _fail(EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT)
        if self.revision_created:
            if previous == current:
                _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
        elif previous is None or previous != current or outcome is not StoryRevalidationOutcome.UNCHANGED:
            _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "previous_revision_id", previous)
        object.__setattr__(self, "current_revision_id", current)
        object.__setattr__(self, "outcome", outcome)


@dataclass(frozen=True, slots=True)
class StoryLifecycleUpdateResult(EngineeringStoryContract):
    memory: EngineeringStoryLifecycleMemory
    revalidation_results: tuple[StoryRevalidationResult, ...]
    diagnostics: tuple[StoryLifecycleDiagnosticCode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.memory, EngineeringStoryLifecycleMemory):
            raise TypeError("memory must be EngineeringStoryLifecycleMemory")
        if (
            isinstance(self.revalidation_results, (str, bytes))
            or not isinstance(self.revalidation_results, Sequence)
            or any(
                not isinstance(item, StoryRevalidationResult)
                for item in self.revalidation_results
            )
        ):
            raise TypeError("revalidation_results contains invalid values")
        if len(self.revalidation_results) > MAX_ENGINEERING_STORY_LIFECYCLE_RESULTS:
            _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
        results = tuple(sorted(
            set(self.revalidation_results),
            key=lambda item: (
                item.project_id.casefold(),
                item.project_id,
                item.canonical_story_id,
                item.current_revision_id,
            ),
        ))
        diagnostics = _stable_enums(
            self.diagnostics,
            StoryLifecycleDiagnosticCode,
            maximum=MAX_ENGINEERING_STORY_LIFECYCLE_DIAGNOSTICS,
            name="diagnostics",
        )
        object.__setattr__(self, "revalidation_results", results)
        object.__setattr__(self, "diagnostics", diagnostics)


def _field_map(record: EngineeringStoryMemoryRecord) -> dict[
    EngineeringStoryFieldName, EngineeringStoryField
]:
    story = record.engineering_story
    return {name: getattr(story, name.value) for name in _FIELD_ORDER}


def _positive(field: EngineeringStoryField) -> bool:
    return field.evidence_state in {
        StoryFieldEvidenceState.CONFIRMED,
        StoryFieldEvidenceState.SUPPORTED,
    }


_SUPPORT_RANK = {
    StoryFieldEvidenceState.UNSUPPORTED: 0,
    StoryFieldEvidenceState.PLAUSIBLE_MISSING: 1,
    StoryFieldEvidenceState.CONFIRMED: 2,
    StoryFieldEvidenceState.SUPPORTED: 3,
}


def _set_delta(
    previous: Sequence[str], current: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    old = set(previous)
    new = set(current)
    return tuple(sorted(new - old)), tuple(sorted(old - new))


def _provenance_delta(
    previous: EngineeringStoryMemoryRecord,
    current: EngineeringStoryMemoryRecord,
) -> StoryRevisionProvenanceDelta:
    old = previous.provenance
    new = current.provenance
    evidence_added, evidence_removed = _set_delta(
        old.evidence_fact_ids, new.evidence_fact_ids
    )
    capability_added, capability_removed = _set_delta(
        old.capability_fact_ids, new.capability_fact_ids
    )
    boundary_added, boundary_removed = _set_delta(
        old.claim_boundary_ids, new.claim_boundary_ids
    )
    lineage_added, lineage_removed = _set_delta(
        old.source_lineage_fingerprints,
        new.source_lineage_fingerprints,
    )
    return StoryRevisionProvenanceDelta(
        added_evidence_fact_ids=evidence_added,
        removed_evidence_fact_ids=evidence_removed,
        added_capability_fact_ids=capability_added,
        removed_capability_fact_ids=capability_removed,
        added_claim_boundary_ids=boundary_added,
        removed_claim_boundary_ids=boundary_removed,
        added_source_lineage_fingerprints=lineage_added,
        removed_source_lineage_fingerprints=lineage_removed,
    )


def _field_changes(
    previous: EngineeringStoryMemoryRecord,
    current: EngineeringStoryMemoryRecord,
) -> tuple[StoryFieldSupportChange, ...]:
    old_fields = _field_map(previous)
    new_fields = _field_map(current)
    results: list[StoryFieldSupportChange] = []
    for name in _FIELD_ORDER:
        old = old_fields[name]
        new = new_fields[name]
        if old == new:
            continue
        changes = {StoryRevisionChangeType.FIELD_CHANGED}
        if not _positive(old) and _positive(new):
            changes.add(StoryRevisionChangeType.FIELD_ADDED)
        if _positive(old) and not _positive(new):
            changes.add(StoryRevisionChangeType.FIELD_REMOVED)
        if _SUPPORT_RANK[new.evidence_state] > _SUPPORT_RANK[old.evidence_state]:
            changes.add(StoryRevisionChangeType.FIELD_SUPPORT_UPGRADED)
        elif _SUPPORT_RANK[new.evidence_state] < _SUPPORT_RANK[old.evidence_state]:
            changes.add(StoryRevisionChangeType.FIELD_SUPPORT_DOWNGRADED)
        evidence_added, evidence_removed = _set_delta(
            old.evidence_fact_ids, new.evidence_fact_ids
        )
        capability_added, capability_removed = _set_delta(
            old.capability_fact_ids, new.capability_fact_ids
        )
        boundary_added, boundary_removed = _set_delta(
            old.claim_boundary_ids, new.claim_boundary_ids
        )
        if boundary_added or boundary_removed:
            changes.add(StoryRevisionChangeType.CLAIM_BOUNDARY_CHANGED)
        results.append(StoryFieldSupportChange(
            field_name=name,
            previous_state=old.evidence_state,
            current_state=new.evidence_state,
            change_types=tuple(changes),
            added_evidence_fact_ids=evidence_added,
            removed_evidence_fact_ids=evidence_removed,
            added_capability_fact_ids=capability_added,
            removed_capability_fact_ids=capability_removed,
            added_claim_boundary_ids=boundary_added,
            removed_claim_boundary_ids=boundary_removed,
        ))
    return tuple(results)


def _record_conflict_fields(
    record: EngineeringStoryMemoryRecord,
) -> tuple[EngineeringStoryFieldName, ...]:
    has_conflict = (
        StoryReconstructionDiagnosticCode.CONFLICTING_FIELDS
        in record.reconstruction_diagnostics
        or SufficiencyDiagnosticCode.FIELD_CONFLICT
        in record.sufficiency_diagnostics
    )
    if not has_conflict:
        return ()
    return tuple(
        name
        for name, field in _field_map(record).items()
        if field.evidence_state is StoryFieldEvidenceState.UNSUPPORTED
        and bool(
            field.evidence_fact_ids
            or field.capability_fact_ids
            or field.claim_boundary_ids
        )
    )


def _boundary_conflict_fields(
    previous: EngineeringStoryMemoryRecord,
    current: EngineeringStoryMemoryRecord,
) -> tuple[EngineeringStoryFieldName, ...]:
    if (
        StoryReconstructionDiagnosticCode.BOUNDARY_RESTRICTIONS
        not in current.reconstruction_diagnostics
    ):
        return ()
    old_fields = _field_map(previous)
    new_fields = _field_map(current)
    return tuple(
        name
        for name in _FIELD_ORDER
        if _positive(old_fields[name])
        and not _positive(new_fields[name])
        and bool(new_fields[name].claim_boundary_ids)
    )


def _support_loss_fields(
    previous: EngineeringStoryMemoryRecord,
    current: EngineeringStoryMemoryRecord,
) -> tuple[EngineeringStoryFieldName, ...]:
    old_fields = _field_map(previous)
    new_fields = _field_map(current)
    affected: list[EngineeringStoryFieldName] = []
    for name in _FIELD_ORDER:
        old = old_fields[name]
        new = new_fields[name]
        if not _positive(old):
            continue
        old_support = set(old.evidence_fact_ids) | set(old.capability_fact_ids)
        new_support = set(new.evidence_fact_ids) | set(new.capability_fact_ids)
        if not _positive(new) or not old_support.issubset(new_support):
            affected.append(name)
    return tuple(affected)


def _viable_technical_core(record: EngineeringStoryMemoryRecord) -> bool:
    fields = _field_map(record)
    return _positive(fields[EngineeringStoryFieldName.MECHANISM]) or _positive(
        fields[EngineeringStoryFieldName.IMPLEMENTATION]
    )


def _validate_new_record(record: EngineeringStoryMemoryRecord) -> None:
    if not isinstance(record, EngineeringStoryMemoryRecord):
        raise TypeError("new_records must contain EngineeringStoryMemoryRecord")
    lifecycle = record.engineering_story.lifecycle
    if (
        lifecycle.status is not EngineeringStoryStatus.ACTIVE
        or lifecycle.requires_revalidation
        or lifecycle.superseded_by_story_id is not None
    ):
        _fail(EngineeringStoryLifecycleErrorCode.UNSAFE_CALLER_LIFECYCLE)


def _change_summary(
    previous: EngineeringStoryMemoryRecord,
    current: EngineeringStoryMemoryRecord,
) -> tuple[
    set[StoryRevisionChangeType],
    StoryRevisionProvenanceDelta,
    tuple[StoryFieldSupportChange, ...],
]:
    changes: set[StoryRevisionChangeType] = set()
    delta = _provenance_delta(previous, current)
    field_changes = _field_changes(previous, current)
    if previous.record_fingerprint != current.record_fingerprint:
        changes.add(StoryRevisionChangeType.RECORD_CONTENT_CHANGED)
    if delta.has_added:
        changes.add(StoryRevisionChangeType.PROVENANCE_ADDED)
    if delta.has_removed:
        changes.add(StoryRevisionChangeType.PROVENANCE_REMOVED)
    if delta.has_added and delta.has_removed:
        changes.add(StoryRevisionChangeType.PROVENANCE_REPLACED)
    for field_change in field_changes:
        changes.update(field_change.change_types)
    old_story = previous.engineering_story
    new_story = current.engineering_story
    if old_story.claim_sufficiency != new_story.claim_sufficiency:
        changes.add(StoryRevisionChangeType.CLAIM_SUFFICIENCY_CHANGED)
    if old_story.story_sufficiency != new_story.story_sufficiency:
        changes.add(StoryRevisionChangeType.STORY_SUFFICIENCY_CHANGED)
    if old_story.opportunity != new_story.opportunity:
        changes.add(StoryRevisionChangeType.OPPORTUNITY_CHANGED)
    if (
        previous.provenance.claim_boundary_ids
        != current.provenance.claim_boundary_ids
    ):
        changes.add(StoryRevisionChangeType.CLAIM_BOUNDARY_CHANGED)
    return changes, delta, field_changes


def _evaluate_lifecycle(
    *,
    canonical_story_id: str,
    previous_revision: EngineeringStoryRevision,
    current_record: EngineeringStoryMemoryRecord,
    changes: set[StoryRevisionChangeType],
    revalidation_requested: bool,
) -> tuple[StoryLifecycleDecision, StoryRevalidationOutcome]:
    previous_lifecycle = previous_revision.lifecycle
    prior_record = previous_revision.record
    conflict_fields = set(_record_conflict_fields(current_record))
    conflict_fields.update(_boundary_conflict_fields(prior_record, current_record))
    support_loss = _support_loss_fields(prior_record, current_record)
    affected = set(support_loss) | conflict_fields
    reasons: set[StoryLifecycleReasonCode] = {
        StoryLifecycleReasonCode.SEMANTIC_STATE_UPDATED
    }
    if conflict_fields:
        changes.add(StoryRevisionChangeType.CONFLICT_INTRODUCED)
        if revalidation_requested:
            changes.add(StoryRevisionChangeType.REVALIDATION_FAILED)
            reasons.add(StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_FAILED)
        reasons.add(StoryLifecycleReasonCode.AUTHORITATIVE_CONFLICT)
        lifecycle = EngineeringStoryLifecycle(
            EngineeringStoryStatus.CONFLICTED,
            requires_revalidation=True,
        )
        outcome = StoryRevalidationOutcome.CONFLICTED
    elif previous_lifecycle.status is EngineeringStoryStatus.CONFLICTED:
        if revalidation_requested:
            changes.update({
                StoryRevisionChangeType.CONFLICT_RESOLVED,
                StoryRevisionChangeType.REVALIDATION_SUCCEEDED,
            })
            reasons.add(StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_SUCCEEDED)
            lifecycle = EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)
            outcome = StoryRevalidationOutcome.UPDATED_ACTIVE
        else:
            lifecycle = previous_lifecycle
            outcome = StoryRevalidationOutcome.CONFLICTED
    elif previous_lifecycle.status is EngineeringStoryStatus.STALE:
        if revalidation_requested and _viable_technical_core(current_record):
            changes.add(StoryRevisionChangeType.REVALIDATION_SUCCEEDED)
            reasons.add(StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_SUCCEEDED)
            lifecycle = EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)
            outcome = StoryRevalidationOutcome.UPDATED_ACTIVE
        else:
            if revalidation_requested:
                changes.add(StoryRevisionChangeType.REVALIDATION_FAILED)
                reasons.add(StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_FAILED)
            reasons.add(StoryLifecycleReasonCode.CORE_SUPPORT_BECAME_STALE)
            lifecycle = EngineeringStoryLifecycle(
                EngineeringStoryStatus.STALE,
                requires_revalidation=True,
            )
            outcome = StoryRevalidationOutcome.STALE
    else:
        critical_loss = any(
            name in {
                EngineeringStoryFieldName.MECHANISM,
                EngineeringStoryFieldName.IMPLEMENTATION,
            }
            for name in support_loss
        ) and not _viable_technical_core(current_record)
        if critical_loss:
            changes.add(StoryRevisionChangeType.REVALIDATION_REQUIRED)
            reasons.add(StoryLifecycleReasonCode.CORE_SUPPORT_BECAME_STALE)
            if revalidation_requested:
                changes.add(StoryRevisionChangeType.REVALIDATION_FAILED)
                reasons.add(StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_FAILED)
            lifecycle = EngineeringStoryLifecycle(
                EngineeringStoryStatus.STALE,
                requires_revalidation=True,
            )
            outcome = StoryRevalidationOutcome.STALE
        elif support_loss or previous_lifecycle.requires_revalidation:
            if revalidation_requested:
                changes.add(StoryRevisionChangeType.REVALIDATION_SUCCEEDED)
                reasons.add(
                    StoryLifecycleReasonCode.EXPLICIT_REVALIDATION_SUCCEEDED
                )
                lifecycle = EngineeringStoryLifecycle(
                    EngineeringStoryStatus.ACTIVE
                )
                outcome = StoryRevalidationOutcome.UPDATED_ACTIVE
            else:
                changes.add(StoryRevisionChangeType.REVALIDATION_REQUIRED)
                reasons.add(
                    StoryLifecycleReasonCode.FIELD_SUPPORT_REQUIRES_REVALIDATION
                )
                lifecycle = EngineeringStoryLifecycle(
                    EngineeringStoryStatus.ACTIVE,
                    requires_revalidation=True,
                )
                outcome = StoryRevalidationOutcome.REQUIRES_REVALIDATION
        else:
            lifecycle = EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)
            outcome = StoryRevalidationOutcome.UPDATED_ACTIVE
            if changes == {
                StoryRevisionChangeType.RECORD_CONTENT_CHANGED,
                StoryRevisionChangeType.OPPORTUNITY_CHANGED,
            } or changes == {StoryRevisionChangeType.OPPORTUNITY_CHANGED}:
                reasons.add(StoryLifecycleReasonCode.OPPORTUNITY_ONLY_UPDATE)
    return StoryLifecycleDecision(
        project_id=current_record.project_id,
        canonical_story_id=canonical_story_id,
        previous_lifecycle=previous_lifecycle,
        resulting_lifecycle=lifecycle,
        reason_codes=tuple(reasons),
        affected_fields=tuple(affected),
    ), outcome


def _make_revision(
    *,
    canonical_story_id: str,
    record: EngineeringStoryMemoryRecord,
    parent_revision_id: str | None,
    changes: Sequence[StoryRevisionChangeType],
    provenance_delta: StoryRevisionProvenanceDelta,
    field_changes: Sequence[StoryFieldSupportChange],
    lifecycle_decision: StoryLifecycleDecision,
    outcome: StoryRevalidationOutcome,
) -> EngineeringStoryRevision:
    revision_id = build_engineering_story_revision_id(
        canonical_story_id=canonical_story_id,
        project_id=record.project_id,
        record=record,
        parent_revision_id=parent_revision_id,
        change_types=changes,
        provenance_delta=provenance_delta,
        field_changes=field_changes,
        lifecycle_decision=lifecycle_decision,
        revalidation_outcome=outcome,
    )
    return EngineeringStoryRevision(
        canonical_story_id=canonical_story_id,
        revision_id=revision_id,
        project_id=record.project_id,
        record=record,
        provenance_fingerprint=record.provenance.provenance_fingerprint,
        record_fingerprint=record.record_fingerprint,
        parent_revision_id=parent_revision_id,
        change_types=tuple(changes),
        provenance_delta=provenance_delta,
        field_changes=tuple(field_changes),
        lifecycle_decision=lifecycle_decision,
        revalidation_outcome=outcome,
    )


def _initial_history(
    *,
    identity: CanonicalEngineeringStoryIdentity,
    record: EngineeringStoryMemoryRecord,
    extra_changes: Sequence[StoryRevisionChangeType],
) -> tuple[EngineeringStoryRevisionHistory, StoryRevalidationResult]:
    conflict_fields = _record_conflict_fields(record)
    changes = {StoryRevisionChangeType.INITIAL_REVISION, *extra_changes}
    if conflict_fields:
        changes.add(StoryRevisionChangeType.CONFLICT_INTRODUCED)
        lifecycle = EngineeringStoryLifecycle(
            EngineeringStoryStatus.CONFLICTED,
            requires_revalidation=True,
        )
        reason = StoryLifecycleReasonCode.INITIALIZED_CONFLICTED
        outcome = StoryRevalidationOutcome.CONFLICTED
    else:
        lifecycle = EngineeringStoryLifecycle(EngineeringStoryStatus.ACTIVE)
        reason = StoryLifecycleReasonCode.INITIALIZED_ACTIVE
        outcome = StoryRevalidationOutcome.VALIDATED_ACTIVE
    reasons = {reason}
    if StoryRevisionChangeType.CANONICAL_SPLIT_RELATION in changes:
        reasons.add(StoryLifecycleReasonCode.CANONICAL_SPLIT_PRESERVED)
    decision = StoryLifecycleDecision(
        project_id=record.project_id,
        canonical_story_id=identity.canonical_story_id,
        previous_lifecycle=None,
        resulting_lifecycle=lifecycle,
        reason_codes=tuple(reasons),
        affected_fields=conflict_fields,
    )
    revision = _make_revision(
        canonical_story_id=identity.canonical_story_id,
        record=record,
        parent_revision_id=None,
        changes=tuple(changes),
        provenance_delta=StoryRevisionProvenanceDelta(),
        field_changes=(),
        lifecycle_decision=decision,
        outcome=outcome,
    )
    history = EngineeringStoryRevisionHistory(
        founding_identity=identity,
        revisions=(revision,),
        current_revision_id=revision.revision_id,
    )
    return history, StoryRevalidationResult(
        project_id=record.project_id,
        canonical_story_id=identity.canonical_story_id,
        previous_revision_id=None,
        current_revision_id=revision.revision_id,
        outcome=outcome,
        revision_created=True,
        lifecycle_decision=decision,
    )


def _update_history(
    *,
    history: EngineeringStoryRevisionHistory,
    record: EngineeringStoryMemoryRecord,
    revalidation_requested: bool,
    extra_changes: Sequence[StoryRevisionChangeType],
) -> tuple[EngineeringStoryRevisionHistory, StoryRevalidationResult]:
    current = history.current_revision
    changes, provenance_delta, field_changes = _change_summary(
        current.record, record
    )
    changes.update(extra_changes)
    decision, outcome = _evaluate_lifecycle(
        canonical_story_id=history.canonical_story_id,
        previous_revision=current,
        current_record=record,
        changes=changes,
        revalidation_requested=revalidation_requested,
    )
    lifecycle_changed = decision.resulting_lifecycle != current.lifecycle
    repeated_failed_revalidation = (
        revalidation_requested
        and current.record_fingerprint == record.record_fingerprint
        and not lifecycle_changed
        and StoryRevisionChangeType.REVALIDATION_FAILED in current.change_types
        and StoryRevisionChangeType.REVALIDATION_FAILED in changes
    )
    if repeated_failed_revalidation:
        changes.clear()
    if not changes and not lifecycle_changed:
        unchanged_decision = StoryLifecycleDecision(
            project_id=history.project_id,
            canonical_story_id=history.canonical_story_id,
            previous_lifecycle=current.lifecycle,
            resulting_lifecycle=current.lifecycle,
            reason_codes=(StoryLifecycleReasonCode.SEMANTIC_STATE_UPDATED,),
        )
        return history, StoryRevalidationResult(
            project_id=history.project_id,
            canonical_story_id=history.canonical_story_id,
            previous_revision_id=current.revision_id,
            current_revision_id=current.revision_id,
            outcome=StoryRevalidationOutcome.UNCHANGED,
            revision_created=False,
            lifecycle_decision=unchanged_decision,
        )
    revision = _make_revision(
        canonical_story_id=history.canonical_story_id,
        record=record,
        parent_revision_id=current.revision_id,
        changes=tuple(changes),
        provenance_delta=provenance_delta,
        field_changes=field_changes,
        lifecycle_decision=decision,
        outcome=outcome,
    )
    updated = EngineeringStoryRevisionHistory(
        founding_identity=history.founding_identity,
        revisions=(*history.revisions, revision),
        current_revision_id=revision.revision_id,
    )
    return updated, StoryRevalidationResult(
        project_id=history.project_id,
        canonical_story_id=history.canonical_story_id,
        previous_revision_id=current.revision_id,
        current_revision_id=revision.revision_id,
        outcome=outcome,
        revision_created=True,
        lifecycle_decision=decision,
    )


def _supersede_history(
    *,
    history: EngineeringStoryRevisionHistory,
    survivor_story_id: str,
) -> tuple[EngineeringStoryRevisionHistory, StoryRevalidationResult | None]:
    current = history.current_revision
    if (
        current.lifecycle.status is EngineeringStoryStatus.SUPERSEDED
        and current.lifecycle.superseded_by_story_id == survivor_story_id
    ):
        return history, None
    lifecycle = EngineeringStoryLifecycle(
        EngineeringStoryStatus.SUPERSEDED,
        superseded_by_story_id=survivor_story_id,
    )
    decision = StoryLifecycleDecision(
        project_id=history.project_id,
        canonical_story_id=history.canonical_story_id,
        previous_lifecycle=current.lifecycle,
        resulting_lifecycle=lifecycle,
        reason_codes=(StoryLifecycleReasonCode.CANONICAL_MERGE_SUPERSEDED,),
    )
    revision = _make_revision(
        canonical_story_id=history.canonical_story_id,
        record=current.record,
        parent_revision_id=current.revision_id,
        changes=(StoryRevisionChangeType.CANONICAL_MERGE_RELATION,),
        provenance_delta=StoryRevisionProvenanceDelta(),
        field_changes=(),
        lifecycle_decision=decision,
        outcome=StoryRevalidationOutcome.SUPERSEDED,
    )
    updated = EngineeringStoryRevisionHistory(
        founding_identity=history.founding_identity,
        revisions=(*history.revisions, revision),
        current_revision_id=revision.revision_id,
    )
    return updated, StoryRevalidationResult(
        project_id=history.project_id,
        canonical_story_id=history.canonical_story_id,
        previous_revision_id=current.revision_id,
        current_revision_id=revision.revision_id,
        outcome=StoryRevalidationOutcome.SUPERSEDED,
        revision_created=True,
        lifecycle_decision=decision,
    )


def _mark_missing_history_stale(
    history: EngineeringStoryRevisionHistory,
) -> tuple[EngineeringStoryRevisionHistory, StoryRevalidationResult | None]:
    """Mark one explicitly absent authoritative candidate safely non-active."""

    current = history.current_revision
    if current.lifecycle.status in {
        EngineeringStoryStatus.STALE,
        EngineeringStoryStatus.CONFLICTED,
        EngineeringStoryStatus.SUPERSEDED,
    }:
        return history, None
    lifecycle = EngineeringStoryLifecycle(
        EngineeringStoryStatus.STALE,
        requires_revalidation=True,
    )
    decision = StoryLifecycleDecision(
        project_id=history.project_id,
        canonical_story_id=history.canonical_story_id,
        previous_lifecycle=current.lifecycle,
        resulting_lifecycle=lifecycle,
        reason_codes=(StoryLifecycleReasonCode.CORE_SUPPORT_BECAME_STALE,),
    )
    revision = _make_revision(
        canonical_story_id=history.canonical_story_id,
        record=current.record,
        parent_revision_id=current.revision_id,
        changes=(StoryRevisionChangeType.REVALIDATION_REQUIRED,),
        provenance_delta=StoryRevisionProvenanceDelta(),
        field_changes=(),
        lifecycle_decision=decision,
        outcome=StoryRevalidationOutcome.STALE,
    )
    updated = EngineeringStoryRevisionHistory(
        founding_identity=history.founding_identity,
        revisions=(*history.revisions, revision),
        current_revision_id=revision.revision_id,
    )
    return updated, StoryRevalidationResult(
        project_id=history.project_id,
        canonical_story_id=history.canonical_story_id,
        previous_revision_id=current.revision_id,
        current_revision_id=revision.revision_id,
        outcome=StoryRevalidationOutcome.STALE,
        revision_created=True,
        lifecycle_decision=decision,
    )


def _validate_identity_history(
    previous: EngineeringStoryIdentityMap,
    current: EngineeringStoryIdentityMap,
) -> None:
    current_identities = {
        item.canonical_story_id: item for item in current.canonical_identities
    }
    for identity in previous.canonical_identities:
        if current_identities.get(identity.canonical_story_id) != identity:
            _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
    current_links = {
        (item.project_id, item.candidate_story_id)
        for item in current.candidate_links
    }
    if any(
        (item.project_id, item.candidate_story_id) not in current_links
        for item in previous.candidate_links
    ):
        _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
    current_alias_ids = {item.alias_story_id for item in current.aliases}
    if any(
        item.alias_story_id not in current_alias_ids for item in previous.aliases
    ):
        _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
    if not set(previous.split_relationships).issubset(
        set(current.split_relationships)
    ):
        _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)


def apply_engineering_story_revisions(
    *,
    canonicalization_result: StoryCanonicalizationResult,
    new_records: Sequence[EngineeringStoryMemoryRecord] = (),
    existing_lifecycle_memory: EngineeringStoryLifecycleMemory | None = None,
    revalidate_canonical_story_ids: Sequence[str] = (),
    missing_canonical_story_ids: Sequence[str] = (),
) -> StoryLifecycleUpdateResult:
    """Apply already-canonicalized snapshots without rematching or retrieval."""

    if not isinstance(canonicalization_result, StoryCanonicalizationResult):
        raise TypeError("canonicalization_result must be StoryCanonicalizationResult")
    identity_map = canonicalization_result.identity_map
    if existing_lifecycle_memory is not None:
        if not isinstance(
            existing_lifecycle_memory, EngineeringStoryLifecycleMemory
        ):
            raise TypeError(
                "existing_lifecycle_memory must be EngineeringStoryLifecycleMemory"
            )
        _validate_identity_history(existing_lifecycle_memory.identity_map, identity_map)
        histories = {
            item.canonical_story_id: item
            for item in existing_lifecycle_memory.histories
        }
        previous_split_relationships = set(
            existing_lifecycle_memory.identity_map.split_relationships
        )
        previous_alias_ids = {
            item.alias_story_id
            for item in existing_lifecycle_memory.identity_map.aliases
        }
    else:
        histories = {}
        previous_split_relationships = set()
        previous_alias_ids = set()
    if isinstance(new_records, (str, bytes)) or not isinstance(
        new_records, Sequence
    ):
        raise TypeError("new_records must be a sequence")
    if len(new_records) > MAX_ENGINEERING_STORY_LIFECYCLE_RESULTS:
        _fail(EngineeringStoryLifecycleErrorCode.BOUND_EXCEEDED)
    by_candidate: dict[tuple[str, str], EngineeringStoryMemoryRecord] = {}
    for record in new_records:
        _validate_new_record(record)
        key = (record.project_id, record.candidate_story_id)
        previous = by_candidate.get(key)
        if previous is not None and previous != record:
            _fail(EngineeringStoryLifecycleErrorCode.CONFLICTING_CANONICAL_HISTORY)
        by_candidate[key] = record
    requested = set(_stable_canonical_ids(revalidate_canonical_story_ids))
    missing = set(_stable_canonical_ids(missing_canonical_story_ids))
    if existing_lifecycle_memory is None and missing:
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    identities = {
        item.canonical_story_id: item for item in identity_map.canonical_identities
    }
    links = {
        (item.project_id, item.candidate_story_id): item.canonical_story_id
        for item in identity_map.candidate_links
    }
    unresolved_candidates = {
        item.candidate_story_id
        for item in canonicalization_result.match_decisions
        if item.canonical_story_id is None
    }
    diagnostics: set[StoryLifecycleDiagnosticCode] = set()
    records_by_canonical: dict[str, EngineeringStoryMemoryRecord] = {}
    for key, record in sorted(by_candidate.items()):
        canonical_id = links.get(key)
        if canonical_id is None:
            if record.candidate_story_id in unresolved_candidates:
                diagnostics.add(
                    StoryLifecycleDiagnosticCode.AMBIGUOUS_IDENTITY_PRESERVED
                )
                continue
            _fail(EngineeringStoryLifecycleErrorCode.UNRESOLVED_CANONICAL_IDENTITY)
        previous = records_by_canonical.get(canonical_id)
        if previous is not None and previous != record:
            _fail(EngineeringStoryLifecycleErrorCode.CONFLICTING_CANONICAL_HISTORY)
        records_by_canonical[canonical_id] = record
    if not requested.issubset(records_by_canonical):
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    if (
        not missing.issubset(histories)
        or missing & set(records_by_canonical)
        or missing & requested
    ):
        _fail(EngineeringStoryLifecycleErrorCode.INVALID_INPUT)
    new_split_relationships = set(identity_map.split_relationships) - (
        previous_split_relationships
    )
    split_candidates = {
        candidate_id
        for relationship in new_split_relationships
        if relationship.outcome is StorySplitOutcome.SPLIT
        for candidate_id in relationship.child_candidate_story_ids
    }
    if any(
        relationship.outcome is StorySplitOutcome.AMBIGUOUS
        for relationship in identity_map.split_relationships
    ):
        diagnostics.add(StoryLifecycleDiagnosticCode.AMBIGUOUS_SPLIT_PRESERVED)
    new_alias_ids = {
        item.alias_story_id for item in identity_map.aliases
    } - previous_alias_ids
    merge_candidates = {
        decision.triggering_candidate_story_id
        for decision in canonicalization_result.merge_decisions
        if decision.outcome is StoryMergeOutcome.MERGED
        and bool(set(decision.aliased_canonical_story_ids) & new_alias_ids)
    }
    results: list[StoryRevalidationResult] = []
    for canonical_id, record in sorted(records_by_canonical.items()):
        identity = identities.get(canonical_id)
        if identity is None or identity.project_id != record.project_id:
            _fail(EngineeringStoryLifecycleErrorCode.CROSS_PROJECT_INPUT)
        extra_changes: set[StoryRevisionChangeType] = set()
        if record.candidate_story_id in split_candidates:
            extra_changes.add(StoryRevisionChangeType.CANONICAL_SPLIT_RELATION)
        if record.candidate_story_id in merge_candidates:
            extra_changes.add(StoryRevisionChangeType.CANONICAL_MERGE_RELATION)
        history = histories.get(canonical_id)
        if history is None:
            history, result = _initial_history(
                identity=identity,
                record=record,
                extra_changes=tuple(extra_changes),
            )
        else:
            history, result = _update_history(
                history=history,
                record=record,
                revalidation_requested=canonical_id in requested,
                extra_changes=tuple(extra_changes),
            )
        histories[canonical_id] = history
        results.append(result)
    for merge in canonicalization_result.merge_decisions:
        if merge.outcome is not StoryMergeOutcome.MERGED:
            continue
        if merge.survivor_canonical_story_id is None:
            _fail(EngineeringStoryLifecycleErrorCode.IDENTITY_HISTORY_MISMATCH)
        for losing_id in merge.aliased_canonical_story_ids:
            history = histories.get(losing_id)
            if history is None:
                continue
            history, result = _supersede_history(
                history=history,
                survivor_story_id=merge.survivor_canonical_story_id,
            )
            histories[losing_id] = history
            if result is not None:
                results.append(result)
    for canonical_id in sorted(missing):
        history, result = _mark_missing_history_stale(histories[canonical_id])
        histories[canonical_id] = history
        if result is not None:
            results.append(result)
            diagnostics.add(
                StoryLifecycleDiagnosticCode.MISSING_AUTHORITATIVE_CANDIDATE_MARKED_STALE
            )
    memory = EngineeringStoryLifecycleMemory(
        identity_map=identity_map,
        histories=tuple(histories.values()),
    )
    return StoryLifecycleUpdateResult(
        memory=memory,
        revalidation_results=tuple(results),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "MAX_ENGINEERING_STORY_LIFECYCLE_DIAGNOSTICS",
    "MAX_ENGINEERING_STORY_LIFECYCLE_RESULTS",
    "MAX_ENGINEERING_STORY_REVISION_HISTORIES",
    "MAX_ENGINEERING_STORY_REVISIONS_PER_CANONICAL",
    "MAX_REVISION_PROVENANCE_DIFF_IDS",
    "EngineeringStoryLifecycleError",
    "EngineeringStoryLifecycleErrorCode",
    "EngineeringStoryLifecycleMemory",
    "EngineeringStoryRevision",
    "EngineeringStoryRevisionHistory",
    "StoryFieldSupportChange",
    "StoryLifecycleDecision",
    "StoryLifecycleDiagnosticCode",
    "StoryLifecycleReasonCode",
    "StoryLifecycleUpdateResult",
    "StoryRevalidationOutcome",
    "StoryRevalidationResult",
    "StoryRevisionChangeType",
    "StoryRevisionProvenanceDelta",
    "apply_engineering_story_revisions",
    "build_engineering_story_revision_id",
]
