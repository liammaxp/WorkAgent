"""Deterministic persistence contracts for evaluated engineering-story candidates.

This module stores bounded snapshots produced by the accepted reconstruction,
sufficiency, and opportunity boundaries.  It does not reconstruct, match,
merge, rank, clarify, retrieve, or assign canonical story identity.

Candidate identity remains derived from exact project authority and the
structural story-cluster event core.  Canonical story identity is deliberately
unresolved until a later matching boundary can reason about merge and split
semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from backend.engineering_story_clustering import (
    StoryCluster,
    StoryClusterIdentityState,
)
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryContract,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    MAX_STORY_PROVENANCE_IDS,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunitySignal,
    StorySufficiency,
    validate_engineering_story_id,
)
from backend.engineering_story_opportunity import (
    StoryOpportunityDetectionResult,
    StoryOpportunityDiagnosticCode,
    StoryOpportunitySignalDecision,
)
from backend.engineering_story_reconstruction import (
    StoryReconstructionDiagnosticCode,
    StoryReconstructionIdentityState,
    StoryReconstructionQuality,
    StoryReconstructionResult,
)
from backend.engineering_story_sufficiency import (
    EngineeringStorySufficiencyResult,
    SufficiencyDiagnosticCode,
)
from backend.project_repository_identity import normalize_project_id


ENGINEERING_STORY_MEMORY_SCHEMA_VERSION = "engineering_story_memory.v1"
MAX_ENGINEERING_STORY_MEMORY_RECORDS = 1_024
MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE = 32 * 1024 * 1024
MAX_SOURCE_LINEAGE_FINGERPRINTS = 256

_CLUSTER_ID_RE = re.compile(r"^story_cluster_([0-9a-f]{24})$")
_CANDIDATE_ID_RE = re.compile(r"^engineering_story_candidate_([0-9a-f]{24})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_SIGNAL_INDEX = {
    value: index for index, value in enumerate(StoryOpportunitySignal)
}


class EngineeringStoryIdentityState(str, Enum):
    PROVISIONAL = "provisional"


class EngineeringStoryMemoryErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    UPSTREAM_RESULT_MISMATCH = "upstream_result_mismatch"
    CROSS_PROJECT_INPUT = "cross_project_input"
    CLUSTER_MISMATCH = "cluster_mismatch"
    PROVISIONAL_IDENTITY_REQUIRED = "provisional_identity_required"
    CANONICAL_IDENTITY_UNRESOLVED = "canonical_identity_unresolved"
    CONFLICTING_CANDIDATE_ID = "conflicting_candidate_id"
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    MAXIMUM_RECORD_COUNT_EXCEEDED = "maximum_record_count_exceeded"
    MAXIMUM_SERIALIZED_SIZE_EXCEEDED = "maximum_serialized_size_exceeded"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    INTEGRITY_FINGERPRINT_MISMATCH = "integrity_fingerprint_mismatch"
    ARTIFACT_MISSING = "artifact_missing"
    MALFORMED_ARTIFACT = "malformed_artifact"
    DESTINATION_IS_DIRECTORY = "destination_is_directory"
    INVALID_EXISTING_ARTIFACT = "invalid_existing_artifact"
    ATOMIC_WRITE_FAILED = "atomic_write_failed"


class EngineeringStoryMemoryError(ValueError):
    """Bounded fail-closed Story Memory contract error."""

    def __init__(self, code: EngineeringStoryMemoryErrorCode | str) -> None:
        self.code = EngineeringStoryMemoryErrorCode(code)
        super().__init__(self.code.value)


class EngineeringStoryMemoryLoadStatus(str, Enum):
    READY = "ready"
    EMPTY = "empty"
    MISSING = "missing"
    INVALID = "invalid"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"


class EngineeringStoryMemoryWriteStatus(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return normalized


def _cluster_id(value: Any) -> str:
    if not isinstance(value, str) or not _CLUSTER_ID_RE.fullmatch(value):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return value


def _verified_fingerprint(provided: Any, expected: str) -> str:
    if provided in (None, ""):
        return expected
    if not isinstance(provided, str) or not _SHA256_RE.fullmatch(provided):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
        )
    if provided != expected:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
        )
    return provided


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
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    index = {item: position for position, item in enumerate(enum_type)}
    normalized = {enum_type(value) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__))


def _authority_ids(
    values: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{kind} must be a sequence")
    if len(values) > MAX_STORY_PROVENANCE_IDS:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    result = tuple(sorted(set(values)))
    for value in result:
        kwargs: dict[str, tuple[str, ...]] = {
            "evidence_fact_ids": (),
            "capability_fact_ids": (),
            "claim_boundary_ids": (),
        }
        kwargs[kind] = (value,)
        EngineeringStoryField(
            value=None,
            evidence_state=StoryFieldEvidenceState.UNSUPPORTED,
            **kwargs,
        )
    return result


def _stable_cluster_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("related_cluster_ids must be a sequence")
    if len(values) > MAX_ENGINEERING_STORY_MEMORY_RECORDS:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return tuple(sorted({_cluster_id(value) for value in values}))


def _stable_digests(
    values: Sequence[str],
    *,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("fingerprints must be a sequence")
    if len(values) > maximum:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return tuple(sorted({_digest(value) for value in values}))


def _identity_payload(identity: "EngineeringStoryIdentity") -> dict[str, Any]:
    return {
        "project_id": identity.project_id,
        "candidate_story_id": identity.candidate_story_id,
        "cluster_id": identity.cluster_id,
        "event_core_fingerprint": identity.event_core_fingerprint,
        "identity_state": identity.identity_state.value,
        "cluster_identity_state": identity.cluster_identity_state.value,
        "reconstruction_identity_state": identity.reconstruction_identity_state.value,
        "canonical_story_id": identity.canonical_story_id,
    }


@dataclass(frozen=True, slots=True)
class EngineeringStoryIdentity(EngineeringStoryContract):
    project_id: str
    candidate_story_id: str
    cluster_id: str
    event_core_fingerprint: str
    identity_state: EngineeringStoryIdentityState
    cluster_identity_state: StoryClusterIdentityState
    reconstruction_identity_state: StoryReconstructionIdentityState
    canonical_story_id: str | None = None
    identity_basis_fingerprint: str = ""

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        candidate_id = validate_engineering_story_id(self.candidate_story_id)
        cluster_id = _cluster_id(self.cluster_id)
        cluster_match = _CLUSTER_ID_RE.fullmatch(cluster_id)
        candidate_match = _CANDIDATE_ID_RE.fullmatch(candidate_id)
        if (
            cluster_match is None
            or candidate_match is None
            or cluster_match.group(1) != candidate_match.group(1)
        ):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.CLUSTER_MISMATCH
            )
        event_core_fingerprint = _digest(self.event_core_fingerprint)
        identity_state = EngineeringStoryIdentityState(self.identity_state)
        cluster_state = StoryClusterIdentityState(self.cluster_identity_state)
        reconstruction_state = StoryReconstructionIdentityState(
            self.reconstruction_identity_state
        )
        if (
            identity_state is not EngineeringStoryIdentityState.PROVISIONAL
            or reconstruction_state
            is not StoryReconstructionIdentityState.PROVISIONAL_CLUSTER_DERIVED
        ):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.PROVISIONAL_IDENTITY_REQUIRED
            )
        if self.canonical_story_id is not None:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.CANONICAL_IDENTITY_UNRESOLVED
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "candidate_story_id", candidate_id)
        object.__setattr__(self, "cluster_id", cluster_id)
        object.__setattr__(self, "event_core_fingerprint", event_core_fingerprint)
        object.__setattr__(self, "identity_state", identity_state)
        object.__setattr__(self, "cluster_identity_state", cluster_state)
        object.__setattr__(self, "reconstruction_identity_state", reconstruction_state)
        expected = _fingerprint(_identity_payload(self))
        object.__setattr__(
            self,
            "identity_basis_fingerprint",
            _verified_fingerprint(self.identity_basis_fingerprint, expected),
        )


def _provenance_payload(
    provenance: "EngineeringStoryMemoryProvenance",
) -> dict[str, Any]:
    return {
        "project_id": provenance.project_id,
        "cluster_id": provenance.cluster_id,
        "event_core_fingerprint": provenance.event_core_fingerprint,
        "evidence_fact_ids": list(provenance.evidence_fact_ids),
        "capability_fact_ids": list(provenance.capability_fact_ids),
        "claim_boundary_ids": list(provenance.claim_boundary_ids),
        "source_lineage_fingerprints": list(
            provenance.source_lineage_fingerprints
        ),
    }


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryProvenance(EngineeringStoryContract):
    project_id: str
    cluster_id: str
    event_core_fingerprint: str
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    claim_boundary_ids: tuple[str, ...]
    source_lineage_fingerprints: tuple[str, ...]
    provenance_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _exact_project_id(self.project_id))
        object.__setattr__(self, "cluster_id", _cluster_id(self.cluster_id))
        object.__setattr__(
            self,
            "event_core_fingerprint",
            _digest(self.event_core_fingerprint),
        )
        object.__setattr__(
            self,
            "evidence_fact_ids",
            _authority_ids(self.evidence_fact_ids, kind="evidence_fact_ids"),
        )
        object.__setattr__(
            self,
            "capability_fact_ids",
            _authority_ids(self.capability_fact_ids, kind="capability_fact_ids"),
        )
        object.__setattr__(
            self,
            "claim_boundary_ids",
            _authority_ids(self.claim_boundary_ids, kind="claim_boundary_ids"),
        )
        object.__setattr__(
            self,
            "source_lineage_fingerprints",
            _stable_digests(
                self.source_lineage_fingerprints,
                maximum=MAX_SOURCE_LINEAGE_FINGERPRINTS,
            ),
        )
        expected = _fingerprint(_provenance_payload(self))
        object.__setattr__(
            self,
            "provenance_fingerprint",
            _verified_fingerprint(self.provenance_fingerprint, expected),
        )


def _record_payload(
    record: "EngineeringStoryMemoryRecord",
    *,
    include_fingerprint: bool,
) -> dict[str, Any]:
    payload = {
        "identity": record.identity.to_dict(),
        "engineering_story": record.engineering_story.to_dict(),
        "provenance": record.provenance.to_dict(),
        "reconstruction_quality": record.reconstruction_quality.value,
        "reconstruction_diagnostics": [
            item.value for item in record.reconstruction_diagnostics
        ],
        "reconstruction_unresolved_fields": [
            item.value for item in record.reconstruction_unresolved_fields
        ],
        "sufficiency_diagnostics": [
            item.value for item in record.sufficiency_diagnostics
        ],
        "opportunity_signal_decisions": [
            item.to_dict() for item in record.opportunity_signal_decisions
        ],
        "opportunity_diagnostics": [
            item.value for item in record.opportunity_diagnostics
        ],
        "related_project_cluster_ids": list(record.related_project_cluster_ids),
    }
    if include_fingerprint:
        payload["record_fingerprint"] = record.record_fingerprint
    return payload


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryRecord(EngineeringStoryContract):
    identity: EngineeringStoryIdentity
    engineering_story: EngineeringStory
    provenance: EngineeringStoryMemoryProvenance
    reconstruction_quality: StoryReconstructionQuality
    reconstruction_diagnostics: tuple[StoryReconstructionDiagnosticCode, ...]
    reconstruction_unresolved_fields: tuple[EngineeringStoryFieldName, ...]
    sufficiency_diagnostics: tuple[SufficiencyDiagnosticCode, ...]
    opportunity_signal_decisions: tuple[StoryOpportunitySignalDecision, ...]
    opportunity_diagnostics: tuple[StoryOpportunityDiagnosticCode, ...]
    related_project_cluster_ids: tuple[str, ...]
    record_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EngineeringStoryIdentity):
            raise TypeError("identity must be an EngineeringStoryIdentity")
        if not isinstance(self.engineering_story, EngineeringStory):
            raise TypeError("engineering_story must be an EngineeringStory")
        if not isinstance(self.provenance, EngineeringStoryMemoryProvenance):
            raise TypeError("provenance must be EngineeringStoryMemoryProvenance")
        story = self.engineering_story
        if (
            story.project_id != self.identity.project_id
            or story.story_id != self.identity.candidate_story_id
            or self.provenance.project_id != self.identity.project_id
            or self.provenance.cluster_id != self.identity.cluster_id
            or self.provenance.event_core_fingerprint
            != self.identity.event_core_fingerprint
        ):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        if (
            story.evidence_fact_ids != self.provenance.evidence_fact_ids
            or story.capability_fact_ids != self.provenance.capability_fact_ids
            or story.claim_boundary_ids != self.provenance.claim_boundary_ids
        ):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        quality = StoryReconstructionQuality(self.reconstruction_quality)
        if quality is StoryReconstructionQuality.BLOCKED:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.INVALID_INPUT
            )
        reconstruction_diagnostics = _stable_enums(
            self.reconstruction_diagnostics,
            StoryReconstructionDiagnosticCode,
            maximum=len(StoryReconstructionDiagnosticCode),
            name="reconstruction_diagnostics",
        )
        unresolved = _stable_enums(
            self.reconstruction_unresolved_fields,
            EngineeringStoryFieldName,
            maximum=len(_FIELD_ORDER),
            name="reconstruction_unresolved_fields",
        )
        expected_unresolved = tuple(
            name
            for name in _FIELD_ORDER
            if not getattr(story, name.value).has_positive_value
        )
        if unresolved != expected_unresolved:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        sufficiency_diagnostics = _stable_enums(
            self.sufficiency_diagnostics,
            SufficiencyDiagnosticCode,
            maximum=len(SufficiencyDiagnosticCode),
            name="sufficiency_diagnostics",
        )
        if (
            isinstance(self.opportunity_signal_decisions, (str, bytes))
            or not isinstance(self.opportunity_signal_decisions, Sequence)
            or any(
                not isinstance(item, StoryOpportunitySignalDecision)
                for item in self.opportunity_signal_decisions
            )
        ):
            raise TypeError(
                "opportunity_signal_decisions must contain signal decisions"
            )
        decisions = tuple(sorted(
            self.opportunity_signal_decisions,
            key=lambda item: _SIGNAL_INDEX[item.signal],
        ))
        if len(decisions) != len({item.signal for item in decisions}):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        expected_signals = tuple(item.signal for item in decisions)
        expected_gaps = story.opportunity.missing_context
        decision_gaps = {
            gap
            for decision in decisions
            for gap in decision.relevant_context_gaps
        }
        if (
            story.opportunity.signals != expected_signals
            or set(expected_gaps) != decision_gaps
        ):
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        opportunity_diagnostics = _stable_enums(
            self.opportunity_diagnostics,
            StoryOpportunityDiagnosticCode,
            maximum=len(StoryOpportunityDiagnosticCode),
            name="opportunity_diagnostics",
        )
        related = _stable_cluster_ids(self.related_project_cluster_ids)
        expected_related = {
            cluster_id
            for decision in decisions
            for cluster_id in decision.related_cluster_ids
        }
        if set(related) != expected_related:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
            )
        for decision in decisions:
            if (
                not set(decision.evidence_fact_ids).issubset(
                    story.evidence_fact_ids
                )
                or not set(decision.capability_fact_ids).issubset(
                    story.capability_fact_ids
                )
            ):
                raise EngineeringStoryMemoryError(
                    EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
                )
        object.__setattr__(self, "reconstruction_quality", quality)
        object.__setattr__(
            self, "reconstruction_diagnostics", reconstruction_diagnostics
        )
        object.__setattr__(self, "reconstruction_unresolved_fields", unresolved)
        object.__setattr__(self, "sufficiency_diagnostics", sufficiency_diagnostics)
        object.__setattr__(self, "opportunity_signal_decisions", decisions)
        object.__setattr__(self, "opportunity_diagnostics", opportunity_diagnostics)
        object.__setattr__(self, "related_project_cluster_ids", related)
        expected = _fingerprint(_record_payload(self, include_fingerprint=False))
        object.__setattr__(
            self,
            "record_fingerprint",
            _verified_fingerprint(self.record_fingerprint, expected),
        )

    @property
    def project_id(self) -> str:
        return self.identity.project_id

    @property
    def cluster_id(self) -> str:
        return self.identity.cluster_id

    @property
    def candidate_story_id(self) -> str:
        return self.identity.candidate_story_id

    @property
    def canonical_story_id(self) -> None:
        return None

    @property
    def claim_sufficiency(self) -> ClaimSufficiency:
        return self.engineering_story.claim_sufficiency

    @property
    def story_sufficiency(self) -> StorySufficiency:
        return self.engineering_story.story_sufficiency

    @property
    def story_opportunity(self) -> StoryOpportunity:
        return self.engineering_story.opportunity


def _memory_payload(
    memory: "EngineeringStoryMemory",
    *,
    include_fingerprint: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": memory.schema_version,
        "records": [record.to_dict() for record in memory.records],
    }
    if include_fingerprint:
        payload["logical_fingerprint"] = memory.logical_fingerprint
    return payload


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemory(EngineeringStoryContract):
    schema_version: str
    records: tuple[EngineeringStoryMemoryRecord, ...]
    logical_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ENGINEERING_STORY_MEMORY_SCHEMA_VERSION:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.UNSUPPORTED_SCHEMA_VERSION
            )
        if isinstance(self.records, (str, bytes)) or not isinstance(
            self.records, Sequence
        ):
            raise TypeError("records must be a sequence")
        if len(self.records) > MAX_ENGINEERING_STORY_MEMORY_RECORDS:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.MAXIMUM_RECORD_COUNT_EXCEEDED
            )
        if any(not isinstance(item, EngineeringStoryMemoryRecord) for item in self.records):
            raise TypeError("records must contain EngineeringStoryMemoryRecord values")
        by_candidate: dict[str, tuple[str, EngineeringStoryMemoryRecord]] = {}
        for record in self.records:
            payload = record.to_json()
            previous = by_candidate.get(record.candidate_story_id)
            if previous is not None and previous[0] != payload:
                raise EngineeringStoryMemoryError(
                    EngineeringStoryMemoryErrorCode.CONFLICTING_CANDIDATE_ID
                )
            by_candidate[record.candidate_story_id] = (payload, record)
        records = tuple(sorted(
            (item[1] for item in by_candidate.values()),
            key=lambda item: (
                item.project_id.casefold(),
                item.project_id,
                item.candidate_story_id,
                item.cluster_id,
            ),
        ))
        object.__setattr__(self, "records", records)
        expected = _fingerprint(_memory_payload(self, include_fingerprint=False))
        object.__setattr__(
            self,
            "logical_fingerprint",
            _verified_fingerprint(self.logical_fingerprint, expected),
        )
        serialized_size = len(_canonical_json_bytes(self.to_dict(), pretty=True)) + 1
        if serialized_size > MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.MAXIMUM_SERIALIZED_SIZE_EXCEEDED
            )


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryLoadResult(EngineeringStoryContract):
    status: EngineeringStoryMemoryLoadStatus
    memory: EngineeringStoryMemory | None
    error_code: EngineeringStoryMemoryErrorCode | None = None

    def __post_init__(self) -> None:
        status = EngineeringStoryMemoryLoadStatus(self.status)
        error = (
            None
            if self.error_code is None
            else EngineeringStoryMemoryErrorCode(self.error_code)
        )
        if status in {
            EngineeringStoryMemoryLoadStatus.READY,
            EngineeringStoryMemoryLoadStatus.EMPTY,
        }:
            if not isinstance(self.memory, EngineeringStoryMemory) or error is not None:
                raise ValueError("successful load requires only a valid memory")
            if (
                status is EngineeringStoryMemoryLoadStatus.EMPTY
            ) != (not self.memory.records):
                raise ValueError("load status must match memory record state")
        elif self.memory is not None or error is None:
            raise ValueError("failed load requires only a bounded error code")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error_code", error)


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryWriteResult(EngineeringStoryContract):
    status: EngineeringStoryMemoryWriteStatus
    logical_fingerprint: str
    bytes_written: int
    previous_artifact_preserved: bool
    round_trip_validated: bool
    error_code: EngineeringStoryMemoryErrorCode | None = None

    def __post_init__(self) -> None:
        status = EngineeringStoryMemoryWriteStatus(self.status)
        fingerprint = _digest(self.logical_fingerprint)
        if (
            isinstance(self.bytes_written, bool)
            or not isinstance(self.bytes_written, int)
            or self.bytes_written < 0
        ):
            raise ValueError("bytes_written must be a non-negative integer")
        if not isinstance(self.previous_artifact_preserved, bool):
            raise TypeError("previous_artifact_preserved must be a boolean")
        if not isinstance(self.round_trip_validated, bool):
            raise TypeError("round_trip_validated must be a boolean")
        error = (
            None
            if self.error_code is None
            else EngineeringStoryMemoryErrorCode(self.error_code)
        )
        if (status is EngineeringStoryMemoryWriteStatus.FAILED) != (error is not None):
            raise ValueError("write status and error_code must agree")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "logical_fingerprint", fingerprint)
        object.__setattr__(self, "error_code", error)


def _event_core_fingerprint(cluster: StoryCluster) -> str:
    return _fingerprint({
        "project_id": cluster.project_id,
        "event_core": cluster.event_core.to_dict(),
        "cluster_identity_state": cluster.identity_state.value,
    })


def _source_lineage_fingerprints(cluster: StoryCluster) -> tuple[str, ...]:
    values = {
        _fingerprint({
            "project_id": cluster.project_id,
            "evidence_fact_id": evidence_input.evidence_fact_id,
            "source_lineage": lineage.to_dict(),
        })
        for evidence_input in cluster.evidence_inputs
        for lineage in evidence_input.source_lineages
    }
    if len(values) > MAX_SOURCE_LINEAGE_FINGERPRINTS:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    return tuple(sorted(values))


def build_engineering_story_memory_record(
    *,
    story_cluster: StoryCluster,
    reconstruction_result: StoryReconstructionResult,
    sufficiency_result: EngineeringStorySufficiencyResult,
    opportunity_result: StoryOpportunityDetectionResult,
) -> EngineeringStoryMemoryRecord:
    """Validate one accepted result chain and build a provisional snapshot."""

    if not isinstance(story_cluster, StoryCluster):
        raise TypeError("story_cluster must be a StoryCluster")
    if not isinstance(reconstruction_result, StoryReconstructionResult):
        raise TypeError("reconstruction_result must be StoryReconstructionResult")
    if not isinstance(sufficiency_result, EngineeringStorySufficiencyResult):
        raise TypeError("sufficiency_result must be EngineeringStorySufficiencyResult")
    if not isinstance(opportunity_result, StoryOpportunityDetectionResult):
        raise TypeError("opportunity_result must be StoryOpportunityDetectionResult")
    project_id = story_cluster.project_id
    if (
        reconstruction_result.project_id != project_id
        or sufficiency_result.project_id != project_id
        or opportunity_result.project_id != project_id
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.CROSS_PROJECT_INPUT
        )
    if (
        reconstruction_result.cluster_id != story_cluster.cluster_id
        or sufficiency_result.cluster_id != story_cluster.cluster_id
        or opportunity_result.cluster_id != story_cluster.cluster_id
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.CLUSTER_MISMATCH
        )
    if (
        sufficiency_result.reconstruction_quality
        is not reconstruction_result.reconstruction_quality
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
        )
    reconstructed_story = reconstruction_result.engineering_story
    sufficiency_story = sufficiency_result.evaluated_story
    opportunity_story = opportunity_result.evaluated_story
    if (
        reconstructed_story is None
        or sufficiency_story is None
        or opportunity_story is None
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INVALID_INPUT
        )
    expected_sufficiency_story = replace(
        reconstructed_story,
        claim_sufficiency=sufficiency_result.claim_sufficiency,
        story_sufficiency=sufficiency_result.story_sufficiency,
    )
    expected_opportunity_story = replace(
        sufficiency_story,
        opportunity=opportunity_result.story_opportunity,
    )
    if (
        sufficiency_story != expected_sufficiency_story
        or opportunity_story != expected_opportunity_story
        or opportunity_result.story_id != reconstructed_story.story_id
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
        )
    if (
        reconstructed_story.evidence_fact_ids
        != story_cluster.member_evidence_fact_ids
        or reconstructed_story.capability_fact_ids
        != story_cluster.member_capability_ids
        or reconstructed_story.claim_boundary_ids
        != story_cluster.claim_boundary_ids
    ):
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.UPSTREAM_RESULT_MISMATCH
        )
    event_core_fingerprint = _event_core_fingerprint(story_cluster)
    identity = EngineeringStoryIdentity(
        project_id=project_id,
        candidate_story_id=reconstructed_story.story_id,
        cluster_id=story_cluster.cluster_id,
        event_core_fingerprint=event_core_fingerprint,
        identity_state=EngineeringStoryIdentityState.PROVISIONAL,
        cluster_identity_state=story_cluster.identity_state,
        reconstruction_identity_state=reconstruction_result.identity_state,
        canonical_story_id=None,
    )
    provenance = EngineeringStoryMemoryProvenance(
        project_id=project_id,
        cluster_id=story_cluster.cluster_id,
        event_core_fingerprint=event_core_fingerprint,
        evidence_fact_ids=opportunity_story.evidence_fact_ids,
        capability_fact_ids=opportunity_story.capability_fact_ids,
        claim_boundary_ids=opportunity_story.claim_boundary_ids,
        source_lineage_fingerprints=_source_lineage_fingerprints(story_cluster),
    )
    return EngineeringStoryMemoryRecord(
        identity=identity,
        engineering_story=opportunity_story,
        provenance=provenance,
        reconstruction_quality=reconstruction_result.reconstruction_quality,
        reconstruction_diagnostics=reconstruction_result.diagnostics,
        reconstruction_unresolved_fields=reconstruction_result.unresolved_fields,
        sufficiency_diagnostics=sufficiency_result.diagnostics,
        opportunity_signal_decisions=opportunity_result.signal_decisions,
        opportunity_diagnostics=opportunity_result.diagnostics,
        related_project_cluster_ids=opportunity_result.related_project_cluster_ids,
    )


def build_engineering_story_memory(
    records: Sequence[EngineeringStoryMemoryRecord] = (),
) -> EngineeringStoryMemory:
    return EngineeringStoryMemory(
        schema_version=ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
        records=tuple(records),
    )


def add_engineering_story_memory_record(
    memory: EngineeringStoryMemory,
    record: EngineeringStoryMemoryRecord,
) -> EngineeringStoryMemory:
    if not isinstance(memory, EngineeringStoryMemory):
        raise TypeError("memory must be an EngineeringStoryMemory")
    if not isinstance(record, EngineeringStoryMemoryRecord):
        raise TypeError("record must be an EngineeringStoryMemoryRecord")
    return build_engineering_story_memory((*memory.records, record))


def get_engineering_story_memory_record(
    memory: EngineeringStoryMemory,
    candidate_story_id: str,
) -> EngineeringStoryMemoryRecord | None:
    if not isinstance(memory, EngineeringStoryMemory):
        raise TypeError("memory must be an EngineeringStoryMemory")
    candidate_id = validate_engineering_story_id(candidate_story_id)
    return next(
        (
            record
            for record in memory.records
            if record.candidate_story_id == candidate_id
        ),
        None,
    )


def replace_engineering_story_memory_record(
    memory: EngineeringStoryMemory,
    record: EngineeringStoryMemoryRecord,
) -> EngineeringStoryMemory:
    if not isinstance(memory, EngineeringStoryMemory):
        raise TypeError("memory must be an EngineeringStoryMemory")
    if not isinstance(record, EngineeringStoryMemoryRecord):
        raise TypeError("record must be an EngineeringStoryMemoryRecord")
    if get_engineering_story_memory_record(memory, record.candidate_story_id) is None:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.CANDIDATE_NOT_FOUND
        )
    return build_engineering_story_memory(tuple(
        record if item.candidate_story_id == record.candidate_story_id else item
        for item in memory.records
    ))


def serialize_engineering_story_memory(memory: EngineeringStoryMemory) -> bytes:
    if not isinstance(memory, EngineeringStoryMemory):
        raise TypeError("memory must be an EngineeringStoryMemory")
    verified = EngineeringStoryMemory.from_dict(memory.to_dict())
    if verified != memory:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
        )
    serialized = _canonical_json_bytes(memory.to_dict(), pretty=True) + b"\n"
    if len(serialized) > MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE:
        raise EngineeringStoryMemoryError(
            EngineeringStoryMemoryErrorCode.MAXIMUM_SERIALIZED_SIZE_EXCEEDED
        )
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


def load_engineering_story_memory(
    path: Path | str,
) -> EngineeringStoryMemoryLoadResult:
    destination = Path(path)
    if not destination.exists():
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.MISSING,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.ARTIFACT_MISSING,
        )
    if destination.is_dir():
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.INVALID,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.DESTINATION_IS_DIRECTORY,
        )
    try:
        if destination.stat().st_size > MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.MAXIMUM_SERIALIZED_SIZE_EXCEEDED
            )
        payload = json.loads(
            destination.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except Exception:
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.INVALID,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.MALFORMED_ARTIFACT,
        )
    if not isinstance(payload, Mapping):
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.INVALID,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.MALFORMED_ARTIFACT,
        )
    if payload.get("schema_version") != ENGINEERING_STORY_MEMORY_SCHEMA_VERSION:
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.UNSUPPORTED_VERSION,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.UNSUPPORTED_SCHEMA_VERSION,
        )
    try:
        memory = EngineeringStoryMemory.from_dict(payload)
    except EngineeringStoryMemoryError as exc:
        status = (
            EngineeringStoryMemoryLoadStatus.INTEGRITY_MISMATCH
            if exc.code
            is EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
            else EngineeringStoryMemoryLoadStatus.INVALID
        )
        return EngineeringStoryMemoryLoadResult(
            status=status,
            memory=None,
            error_code=exc.code,
        )
    except (TypeError, ValueError):
        return EngineeringStoryMemoryLoadResult(
            status=EngineeringStoryMemoryLoadStatus.INVALID,
            memory=None,
            error_code=EngineeringStoryMemoryErrorCode.MALFORMED_ARTIFACT,
        )
    return EngineeringStoryMemoryLoadResult(
        status=(
            EngineeringStoryMemoryLoadStatus.EMPTY
            if not memory.records
            else EngineeringStoryMemoryLoadStatus.READY
        ),
        memory=memory,
    )


def _replace_staged_engineering_story_memory(
    staged: Path,
    destination: Path,
) -> None:
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


def write_engineering_story_memory(
    path: Path | str,
    memory: EngineeringStoryMemory,
) -> EngineeringStoryMemoryWriteResult:
    """Round-trip one explicit-path artifact before atomic replacement."""

    if not isinstance(memory, EngineeringStoryMemory):
        raise TypeError("memory must be an EngineeringStoryMemory")
    destination = Path(path)
    serialized = serialize_engineering_story_memory(memory)
    existed = destination.exists()
    if destination.is_dir():
        return EngineeringStoryMemoryWriteResult(
            EngineeringStoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            True,
            False,
            EngineeringStoryMemoryErrorCode.DESTINATION_IS_DIRECTORY,
        )
    if existed:
        current = load_engineering_story_memory(destination)
        if current.status not in {
            EngineeringStoryMemoryLoadStatus.READY,
            EngineeringStoryMemoryLoadStatus.EMPTY,
        } or current.memory is None:
            return EngineeringStoryMemoryWriteResult(
                EngineeringStoryMemoryWriteStatus.FAILED,
                memory.logical_fingerprint,
                0,
                True,
                False,
                EngineeringStoryMemoryErrorCode.INVALID_EXISTING_ARTIFACT,
            )
        if current.memory == memory:
            return EngineeringStoryMemoryWriteResult(
                EngineeringStoryMemoryWriteStatus.UNCHANGED,
                memory.logical_fingerprint,
                0,
                True,
                True,
            )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return EngineeringStoryMemoryWriteResult(
            EngineeringStoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            existed,
            False,
            EngineeringStoryMemoryErrorCode.ATOMIC_WRITE_FAILED,
        )
    staged: Path | None = None
    try:
        descriptor, staged_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".stage",
        )
        staged = Path(staged_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        staged_load = load_engineering_story_memory(staged)
        expected_status = (
            EngineeringStoryMemoryLoadStatus.EMPTY
            if not memory.records
            else EngineeringStoryMemoryLoadStatus.READY
        )
        if staged_load.status is not expected_status or staged_load.memory != memory:
            raise EngineeringStoryMemoryError(
                EngineeringStoryMemoryErrorCode.INTEGRITY_FINGERPRINT_MISMATCH
            )
        _replace_staged_engineering_story_memory(staged, destination)
        staged = None
        _sync_parent_directory(destination.parent)
        return EngineeringStoryMemoryWriteResult(
            EngineeringStoryMemoryWriteStatus.UPDATED
            if existed
            else EngineeringStoryMemoryWriteStatus.CREATED,
            memory.logical_fingerprint,
            len(serialized),
            existed,
            True,
        )
    except Exception:
        return EngineeringStoryMemoryWriteResult(
            EngineeringStoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            existed,
            False,
            EngineeringStoryMemoryErrorCode.ATOMIC_WRITE_FAILED,
        )
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "ENGINEERING_STORY_MEMORY_SCHEMA_VERSION",
    "MAX_ENGINEERING_STORY_MEMORY_RECORDS",
    "MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE",
    "MAX_SOURCE_LINEAGE_FINGERPRINTS",
    "EngineeringStoryIdentity",
    "EngineeringStoryIdentityState",
    "EngineeringStoryMemory",
    "EngineeringStoryMemoryError",
    "EngineeringStoryMemoryErrorCode",
    "EngineeringStoryMemoryLoadResult",
    "EngineeringStoryMemoryLoadStatus",
    "EngineeringStoryMemoryProvenance",
    "EngineeringStoryMemoryRecord",
    "EngineeringStoryMemoryWriteResult",
    "EngineeringStoryMemoryWriteStatus",
    "add_engineering_story_memory_record",
    "build_engineering_story_memory",
    "build_engineering_story_memory_record",
    "get_engineering_story_memory_record",
    "load_engineering_story_memory",
    "replace_engineering_story_memory_record",
    "serialize_engineering_story_memory",
    "write_engineering_story_memory",
]
