"""Deterministic opportunity detection for evidence-grounded engineering stories.

Opportunity describes intrinsic, recoverable story value.  It is not hiring
relevance, question value, retrieval routing, resume priority, or persistence.
Only accepted reconstruction, sufficiency, structural cluster, and optional
same-project cluster metadata are consumed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any

from backend.engineering_story_clustering import (
    MAX_STORY_CLUSTERS,
    StoryCluster,
    StoryClusterLineageState,
    StoryClusterQuality,
)
from backend.engineering_story_evidence import (
    CapabilityLineageState,
    StoryEvidenceInput,
)
from backend.engineering_story_models import (
    EngineeringStory,
    EngineeringStoryContract,
    EngineeringStoryField,
    EngineeringStoryFieldName,
    EngineeringStoryType,
    MAX_STORY_PROVENANCE_IDS,
    StoryContextGap,
    StoryFieldEvidenceState,
    StoryOpportunity,
    StoryOpportunityLevel,
    StoryOpportunitySignal,
    SufficiencyLevel,
)
from backend.engineering_story_reconstruction import (
    StoryReconstructionQuality,
    StoryReconstructionResult,
)
from backend.engineering_story_sufficiency import (
    EngineeringStorySufficiencyResult,
)
from backend.project_evidence_models import EvidenceStatus, EvidenceType
from backend.project_repository_identity import normalize_project_id


MAX_OPPORTUNITY_DIAGNOSTICS = 8
MAX_RELATED_PROJECT_CLUSTERS = MAX_STORY_CLUSTERS - 1
MAX_STRUCTURAL_SUBSYSTEM_KEYS = 32
MAX_CAPABILITY_TYPES_PER_CONTEXT = 16

_CLUSTER_ID_RE = re.compile(r"^story_cluster_[0-9a-f]{24}$")
_SUBSYSTEM_KEY_RE = re.compile(r"^subsystem_[0-9a-f]{24}$")
_CAPABILITY_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{1,99}$")

_FIELD_ORDER = tuple(EngineeringStoryFieldName)
_SIGNAL_INDEX = {value: index for index, value in enumerate(StoryOpportunitySignal)}


class StoryOpportunitySignalStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class StoryOpportunityReasonCode(str, Enum):
    ARCHITECTURE_EVENT_RECONSTRUCTED = "architecture_event_reconstructed"
    ARCHITECTURE_EVENT_WITH_MISSING_DECISION = (
        "architecture_event_with_missing_decision"
    )
    DEFENSIVE_MECHANISM_CLUSTER = "defensive_mechanism_cluster"
    SPECIFIC_FAILURE_WITH_VALIDATION = "specific_failure_with_validation"
    REPEATED_SUBSYSTEM_EVENTS = "repeated_subsystem_events"
    SIGNIFICANT_DESIGN_CONTEXT_MISSING = "significant_design_context_missing"
    MEANINGFUL_EVENT_WITH_MISSING_HUMAN_CONTEXT = (
        "meaningful_event_with_missing_human_context"
    )


class StoryOpportunityDiagnosticCode(str, Enum):
    NO_MEANINGFUL_EVENT_SIGNAL = "no_meaningful_event_signal"
    NO_RELEVANT_CONTEXT_GAP = "no_relevant_context_gap"
    BLOCKED_RECONSTRUCTION = "blocked_reconstruction"
    AMBIGUOUS_RECONSTRUCTION_LIMIT = "ambiguous_reconstruction_limit"
    MINIMAL_RECONSTRUCTION_LIMIT = "minimal_reconstruction_limit"
    WEAK_CLAIM_AUTHORITY_LIMIT = "weak_claim_authority_limit"
    REPEATED_SUBSYSTEM_CONTEXT_USED = "repeated_subsystem_context_used"


class StoryOpportunityDetectionErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    CROSS_PROJECT_INPUT = "cross_project_input"
    CLUSTER_MISMATCH = "cluster_mismatch"
    STORY_MISMATCH = "story_mismatch"
    INVALID_PROJECT_CONTEXT = "invalid_project_context"
    BOUND_EXCEEDED = "bound_exceeded"


class StoryOpportunityDetectionError(ValueError):
    """Bounded fail-closed error for inconsistent opportunity inputs."""

    def __init__(
        self,
        code: StoryOpportunityDetectionErrorCode | str,
        reference_id: str | None = None,
    ) -> None:
        self.code = StoryOpportunityDetectionErrorCode(code)
        self.reference_id = _bounded_reference(reference_id)
        message = self.code.value
        if self.reference_id is not None:
            message += f":{self.reference_id}"
        super().__init__(message)


_REASON_BY_SIGNAL = {
    StoryOpportunitySignal.ARCHITECTURE_MIGRATION: frozenset({
        StoryOpportunityReasonCode.ARCHITECTURE_EVENT_RECONSTRUCTED,
        StoryOpportunityReasonCode.ARCHITECTURE_EVENT_WITH_MISSING_DECISION,
    }),
    StoryOpportunitySignal.DEFENSIVE_ENGINEERING_CLUSTER: frozenset({
        StoryOpportunityReasonCode.DEFENSIVE_MECHANISM_CLUSTER,
    }),
    StoryOpportunitySignal.FAILURE_SPECIFIC_TEST_CLUSTER: frozenset({
        StoryOpportunityReasonCode.SPECIFIC_FAILURE_WITH_VALIDATION,
    }),
    StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING: frozenset({
        StoryOpportunityReasonCode.REPEATED_SUBSYSTEM_EVENTS,
    }),
    StoryOpportunitySignal.MAJOR_DESIGN_DECISION: frozenset({
        StoryOpportunityReasonCode.SIGNIFICANT_DESIGN_CONTEXT_MISSING,
    }),
    StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT: frozenset({
        StoryOpportunityReasonCode.MEANINGFUL_EVENT_WITH_MISSING_HUMAN_CONTEXT,
    }),
}

_DEFENSIVE_CAPABILITY_TYPES = frozenset({
    "claim_validation",
    "failure_recovery",
    "latex_validation_and_repair",
    "output_quality_control",
    "template_pollution_blocking",
    "test_and_regression_hardening",
})
_DESIGN_STORY_TYPES = frozenset({
    EngineeringStoryType.ARCHITECTURE_CHANGE,
    EngineeringStoryType.RETRIEVAL_REDESIGN,
    EngineeringStoryType.DATA_OR_MEMORY_SYSTEM,
    EngineeringStoryType.WORKFLOW_AUTOMATION,
})
_DESIGN_EVIDENCE_TYPES = frozenset({
    EvidenceType.ARCHITECTURE,
    EvidenceType.RETRIEVAL,
    EvidenceType.DATA_PERSISTENCE,
    EvidenceType.WORKFLOW,
})
_HARDENING_EVIDENCE_TYPES = frozenset({
    EvidenceType.BUG_FIX,
    EvidenceType.FAILURE_RECOVERY,
    EvidenceType.TESTING,
    EvidenceType.VALIDATION,
})
_DESIGN_GAPS = frozenset({
    StoryContextGap.DECISION_REASON,
    StoryContextGap.TRADEOFF,
})
_FAILURE_GAPS = frozenset({
    StoryContextGap.PROBLEM_CONTEXT,
    StoryContextGap.TRIGGER,
    StoryContextGap.OBSERVABLE_OUTCOME_CONTEXT,
})
_HUMAN_GAPS = frozenset({
    StoryContextGap.PROBLEM_CONTEXT,
    StoryContextGap.TRIGGER,
    StoryContextGap.OWNERSHIP,
    StoryContextGap.STAKEHOLDER_CONTEXT,
})
_AFFIRMATIVE_EVIDENCE_STATUSES = frozenset({
    EvidenceStatus.ACCEPTED,
    EvidenceStatus.SUPPORTING,
})


def _bounded_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value if character.isprintable())
    return normalized[:300] or None


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.INVALID_INPUT
        )
    return normalized


def _cluster_id(value: Any) -> str:
    if not isinstance(value, str) or not _CLUSTER_ID_RE.fullmatch(value):
        raise ValueError("cluster_id must be a canonical story-cluster ID")
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
        raise ValueError(f"{name} exceeds maximum item count {maximum}")
    index = {value: position for position, value in enumerate(enum_type)}
    return tuple(sorted({enum_type(value) for value in values}, key=index.__getitem__))


def _stable_ids(
    values: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{kind} must be a sequence")
    if len(values) > MAX_STORY_PROVENANCE_IDS:
        raise ValueError(f"{kind} exceeds maximum item count")
    normalized = tuple(sorted(set(values)))
    for value in normalized:
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
    return normalized


def _stable_cluster_ids(
    values: Sequence[str],
    *,
    maximum: int = MAX_RELATED_PROJECT_CLUSTERS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("related_cluster_ids must be a sequence")
    if len(values) > maximum:
        raise ValueError("related_cluster_ids exceed the bounded context")
    return tuple(sorted({_cluster_id(value) for value in values}))


@dataclass(frozen=True, slots=True)
class StoryOpportunitySignalDecision(EngineeringStoryContract):
    signal: StoryOpportunitySignal
    strength: StoryOpportunitySignalStrength
    supporting_story_fields: tuple[EngineeringStoryFieldName, ...]
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    relevant_context_gaps: tuple[StoryContextGap, ...]
    related_cluster_ids: tuple[str, ...]
    reason_code: StoryOpportunityReasonCode

    def __post_init__(self) -> None:
        signal = StoryOpportunitySignal(self.signal)
        strength = StoryOpportunitySignalStrength(self.strength)
        fields = _stable_enums(
            self.supporting_story_fields,
            EngineeringStoryFieldName,
            maximum=len(_FIELD_ORDER),
            name="supporting_story_fields",
        )
        evidence_ids = _stable_ids(self.evidence_fact_ids, kind="evidence_fact_ids")
        capability_ids = _stable_ids(
            self.capability_fact_ids,
            kind="capability_fact_ids",
        )
        gaps = _stable_enums(
            self.relevant_context_gaps,
            StoryContextGap,
            maximum=len(StoryContextGap),
            name="relevant_context_gaps",
        )
        related = _stable_cluster_ids(self.related_cluster_ids)
        reason = StoryOpportunityReasonCode(self.reason_code)
        if reason not in _REASON_BY_SIGNAL[signal]:
            raise ValueError("opportunity reason does not match its signal")
        if not evidence_ids:
            raise ValueError("a detected signal requires current-story evidence")
        if signal is StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING:
            if not related:
                raise ValueError("repeated-subsystem signal requires related clusters")
        elif related:
            raise ValueError("only repeated-subsystem signal may retain related clusters")
        if signal in {
            StoryOpportunitySignal.MAJOR_DESIGN_DECISION,
            StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT,
        } and not gaps:
            raise ValueError("context-gap opportunity signal requires a gap")
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "supporting_story_fields", fields)
        object.__setattr__(self, "evidence_fact_ids", evidence_ids)
        object.__setattr__(self, "capability_fact_ids", capability_ids)
        object.__setattr__(self, "relevant_context_gaps", gaps)
        object.__setattr__(self, "related_cluster_ids", related)
        object.__setattr__(self, "reason_code", reason)


@dataclass(frozen=True, slots=True)
class StoryOpportunityClusterContext(EngineeringStoryContract):
    project_id: str
    cluster_id: str
    structural_subsystem_keys: tuple[str, ...]
    evidence_types: tuple[EvidenceType, ...]
    capability_types: tuple[str, ...]
    quality: StoryClusterQuality
    lineage_state: StoryClusterLineageState

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        cluster_id = _cluster_id(self.cluster_id)
        if (
            isinstance(self.structural_subsystem_keys, (str, bytes))
            or not isinstance(self.structural_subsystem_keys, Sequence)
        ):
            raise TypeError("structural_subsystem_keys must be a sequence")
        if len(self.structural_subsystem_keys) > MAX_STRUCTURAL_SUBSYSTEM_KEYS:
            raise ValueError("structural_subsystem_keys exceed the bounded context")
        subsystem_keys = tuple(sorted(set(self.structural_subsystem_keys)))
        if any(not _SUBSYSTEM_KEY_RE.fullmatch(value) for value in subsystem_keys):
            raise ValueError("invalid structural subsystem key")
        evidence_types = _stable_enums(
            self.evidence_types,
            EvidenceType,
            maximum=len(EvidenceType),
            name="evidence_types",
        )
        if not evidence_types:
            raise ValueError("cluster context requires evidence types")
        if (
            isinstance(self.capability_types, (str, bytes))
            or not isinstance(self.capability_types, Sequence)
        ):
            raise TypeError("capability_types must be a sequence")
        if len(self.capability_types) > MAX_CAPABILITY_TYPES_PER_CONTEXT:
            raise ValueError("capability_types exceed the bounded context")
        capability_types = tuple(sorted(set(self.capability_types)))
        if any(not _CAPABILITY_TYPE_RE.fullmatch(value) for value in capability_types):
            raise ValueError("invalid capability type in project context")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "cluster_id", cluster_id)
        object.__setattr__(self, "structural_subsystem_keys", subsystem_keys)
        object.__setattr__(self, "evidence_types", evidence_types)
        object.__setattr__(self, "capability_types", capability_types)
        object.__setattr__(self, "quality", StoryClusterQuality(self.quality))
        object.__setattr__(
            self,
            "lineage_state",
            StoryClusterLineageState(self.lineage_state),
        )


@dataclass(frozen=True, slots=True)
class StoryOpportunityProjectContext(EngineeringStoryContract):
    project_id: str
    clusters: tuple[StoryOpportunityClusterContext, ...] = ()

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        if isinstance(self.clusters, (str, bytes)) or not isinstance(
            self.clusters, Sequence
        ):
            raise TypeError("clusters must be a sequence")
        if len(self.clusters) > MAX_STORY_CLUSTERS:
            raise ValueError("clusters exceed the bounded project context")
        if any(not isinstance(item, StoryOpportunityClusterContext) for item in self.clusters):
            raise TypeError("clusters must contain StoryOpportunityClusterContext values")
        clusters = tuple(sorted(self.clusters, key=lambda item: item.cluster_id))
        if any(item.project_id != project_id for item in clusters):
            raise StoryOpportunityDetectionError(
                StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT
            )
        ids = [item.cluster_id for item in clusters]
        if len(ids) != len(set(ids)):
            raise ValueError("project context contains duplicate cluster IDs")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "clusters", clusters)


@dataclass(frozen=True, slots=True)
class StoryOpportunityDetectionResult(EngineeringStoryContract):
    cluster_id: str
    project_id: str
    story_id: str | None
    evaluated_story: EngineeringStory | None
    story_opportunity: StoryOpportunity
    signal_decisions: tuple[StoryOpportunitySignalDecision, ...]
    relevant_context_gaps: tuple[StoryContextGap, ...]
    related_project_cluster_ids: tuple[str, ...]
    diagnostics: tuple[StoryOpportunityDiagnosticCode, ...]

    def __post_init__(self) -> None:
        cluster_id = _cluster_id(self.cluster_id)
        project_id = _exact_project_id(self.project_id)
        if not isinstance(self.story_opportunity, StoryOpportunity):
            raise TypeError("story_opportunity must be a StoryOpportunity")
        if (
            isinstance(self.signal_decisions, (str, bytes))
            or not isinstance(self.signal_decisions, Sequence)
            or any(
                not isinstance(item, StoryOpportunitySignalDecision)
                for item in self.signal_decisions
            )
        ):
            raise TypeError("signal_decisions must contain signal decisions")
        decisions = tuple(sorted(
            self.signal_decisions,
            key=lambda item: _SIGNAL_INDEX[item.signal],
        ))
        if len(decisions) != len({item.signal for item in decisions}):
            raise ValueError("signal_decisions contain duplicate signals")
        gaps = _stable_enums(
            self.relevant_context_gaps,
            StoryContextGap,
            maximum=len(StoryContextGap),
            name="relevant_context_gaps",
        )
        related = _stable_cluster_ids(self.related_project_cluster_ids)
        diagnostics = _stable_enums(
            self.diagnostics,
            StoryOpportunityDiagnosticCode,
            maximum=MAX_OPPORTUNITY_DIAGNOSTICS,
            name="diagnostics",
        )
        expected_signals = tuple(item.signal for item in decisions)
        expected_gaps = _stable_enums(
            tuple({
                gap
                for item in decisions
                for gap in item.relevant_context_gaps
            }),
            StoryContextGap,
            maximum=len(StoryContextGap),
            name="decision_context_gaps",
        )
        expected_related = _stable_cluster_ids(tuple(
            cluster_id_value
            for item in decisions
            for cluster_id_value in item.related_cluster_ids
        ))
        if self.story_opportunity.signals != expected_signals:
            raise ValueError("StoryOpportunity signals must match signal decisions")
        if self.story_opportunity.missing_context != expected_gaps or gaps != expected_gaps:
            raise ValueError("opportunity context gaps must match signal decisions")
        if related != expected_related:
            raise ValueError("related project clusters must match signal decisions")
        story = self.evaluated_story
        if story is None:
            if self.story_id is not None or decisions or gaps or related:
                raise ValueError("missing story cannot retain opportunity provenance")
            if self.story_opportunity.level is not StoryOpportunityLevel.NONE:
                raise ValueError("missing story requires no opportunity")
        else:
            if not isinstance(story, EngineeringStory):
                raise TypeError("evaluated_story must be an EngineeringStory or None")
            if story.project_id != project_id or story.story_id != self.story_id:
                raise ValueError("evaluated story identity must match result")
            if story.opportunity != self.story_opportunity:
                raise ValueError("evaluated story opportunity must match result")
            for decision in decisions:
                if not set(decision.evidence_fact_ids).issubset(
                    story.evidence_fact_ids
                ):
                    raise ValueError("signal evidence must belong to the current story")
                if not set(decision.capability_fact_ids).issubset(
                    story.capability_fact_ids
                ):
                    raise ValueError("signal capabilities must belong to the current story")
                if any(
                    not getattr(story, field_name.value).has_positive_value
                    for field_name in decision.supporting_story_fields
                ):
                    raise ValueError("signal fields must have positive story evidence")
        object.__setattr__(self, "cluster_id", cluster_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "signal_decisions", decisions)
        object.__setattr__(self, "relevant_context_gaps", gaps)
        object.__setattr__(self, "related_project_cluster_ids", related)
        object.__setattr__(self, "diagnostics", diagnostics)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _subsystem_key(kind: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256(
        _canonical_json({"kind": kind, "values": list(values)}).encode("utf-8")
    ).hexdigest()[:24]
    return f"subsystem_{digest}"


def _affirmative_inputs(cluster: StoryCluster) -> tuple[StoryEvidenceInput, ...]:
    return tuple(
        item
        for item in cluster.evidence_inputs
        if item.evidence_status in _AFFIRMATIVE_EVIDENCE_STATUSES
    )


def _cluster_subsystem_keys(cluster: StoryCluster) -> tuple[str, ...]:
    keys: set[str] = set()
    for evidence_input in _affirmative_inputs(cluster):
        for lineage in evidence_input.source_lineages:
            if lineage.repository and lineage.file_path:
                keys.add(_subsystem_key(
                    "repository_path",
                    (lineage.repository, lineage.file_path),
                ))
            if lineage.repository and lineage.symbol:
                keys.add(_subsystem_key(
                    "repository_symbol",
                    (lineage.repository, lineage.symbol),
                ))
            keys.add(_subsystem_key(
                "source_identity",
                (lineage.source_type, lineage.source_id),
            ))
    if len(keys) > MAX_STRUCTURAL_SUBSYSTEM_KEYS:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.BOUND_EXCEEDED,
            cluster.cluster_id,
        )
    return tuple(sorted(keys))


def _cluster_context(
    cluster: StoryCluster,
) -> StoryOpportunityClusterContext | None:
    affirmative_inputs = _affirmative_inputs(cluster)
    if not affirmative_inputs:
        return None
    affirmative_ids = {item.evidence_fact_id for item in affirmative_inputs}
    return StoryOpportunityClusterContext(
        project_id=cluster.project_id,
        cluster_id=cluster.cluster_id,
        structural_subsystem_keys=_cluster_subsystem_keys(cluster),
        evidence_types=tuple(item.evidence_type for item in affirmative_inputs),
        capability_types=tuple(
            item.capability_type
            for item in cluster.capability_lineages
            if item.present
            and item.state is CapabilityLineageState.RESOLVED
            and set(item.source_evidence_fact_ids).issubset(affirmative_ids)
        ),
        quality=cluster.quality,
        lineage_state=cluster.lineage_state,
    )


def build_story_opportunity_project_context(
    *,
    project_id: str,
    clusters: Sequence[StoryCluster],
) -> StoryOpportunityProjectContext:
    """Reduce accepted same-project clusters to bounded, path-free metadata."""

    canonical_project = _exact_project_id(project_id)
    if isinstance(clusters, (str, bytes)) or not isinstance(clusters, Sequence):
        raise TypeError("clusters must be a sequence")
    if len(clusters) > MAX_STORY_CLUSTERS:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.BOUND_EXCEEDED
        )
    if any(not isinstance(cluster, StoryCluster) for cluster in clusters):
        raise TypeError("clusters must contain StoryCluster values")
    if any(cluster.project_id != canonical_project for cluster in clusters):
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT
        )
    ids = [cluster.cluster_id for cluster in clusters]
    if len(ids) != len(set(ids)):
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.INVALID_PROJECT_CONTEXT
        )
    return StoryOpportunityProjectContext(
        project_id=canonical_project,
        clusters=tuple(
            context
            for cluster in clusters
            if (context := _cluster_context(cluster)) is not None
        ),
    )


def _field_names(
    story: EngineeringStory,
    names: Sequence[EngineeringStoryFieldName],
) -> tuple[EngineeringStoryFieldName, ...]:
    return tuple(name for name in names if getattr(story, name.value).has_positive_value)


def _field_provenance(
    story: EngineeringStory,
    names: Sequence[EngineeringStoryFieldName],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    evidence_ids = tuple(sorted({
        evidence_id
        for name in names
        for evidence_id in getattr(story, name.value).evidence_fact_ids
    }))
    capability_ids = tuple(sorted({
        capability_id
        for name in names
        for capability_id in getattr(story, name.value).capability_fact_ids
    }))
    return evidence_ids, capability_ids


def _typed_evidence_ids(
    cluster: StoryCluster,
    evidence_types: frozenset[EvidenceType],
) -> tuple[str, ...]:
    return tuple(
        item.evidence_fact_id
        for item in cluster.evidence_inputs
        if item.evidence_status in _AFFIRMATIVE_EVIDENCE_STATUSES
        and item.evidence_type in evidence_types
    )


def _capability_ids(
    cluster: StoryCluster,
    capability_types: frozenset[str],
) -> tuple[str, ...]:
    affirmative_ids = {
        item.evidence_fact_id for item in _affirmative_inputs(cluster)
    }
    return tuple(
        item.capability_id
        for item in cluster.capability_lineages
        if item.present
        and item.state is CapabilityLineageState.RESOLVED
        and item.capability_type in capability_types
        and set(item.source_evidence_fact_ids).issubset(affirmative_ids)
    )


def _gaps(
    available: Sequence[StoryContextGap],
    relevant: frozenset[StoryContextGap],
) -> tuple[StoryContextGap, ...]:
    return tuple(gap for gap in StoryContextGap if gap in available and gap in relevant)


def _signal_decision(
    *,
    signal: StoryOpportunitySignal,
    strength: StoryOpportunitySignalStrength,
    story: EngineeringStory,
    supporting_fields: Sequence[EngineeringStoryFieldName],
    evidence_fact_ids: Sequence[str],
    capability_fact_ids: Sequence[str] = (),
    relevant_context_gaps: Sequence[StoryContextGap] = (),
    related_cluster_ids: Sequence[str] = (),
    reason_code: StoryOpportunityReasonCode,
) -> StoryOpportunitySignalDecision:
    evidence_ids = tuple(sorted(set(evidence_fact_ids)))
    capability_ids = tuple(sorted(set(capability_fact_ids)))
    if not evidence_ids:
        evidence_ids, field_capabilities = _field_provenance(
            story,
            supporting_fields,
        )
        capability_ids = tuple(sorted(set(capability_ids) | set(field_capabilities)))
    return StoryOpportunitySignalDecision(
        signal=signal,
        strength=strength,
        supporting_story_fields=tuple(supporting_fields),
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        relevant_context_gaps=tuple(relevant_context_gaps),
        related_cluster_ids=tuple(related_cluster_ids),
        reason_code=reason_code,
    )


def _architecture_signal(
    story: EngineeringStory,
    cluster: StoryCluster,
    context_gaps: Sequence[StoryContextGap],
) -> StoryOpportunitySignalDecision | None:
    fields = _field_names(story, (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    ))
    evidence_ids = _typed_evidence_ids(cluster, frozenset({EvidenceType.ARCHITECTURE}))
    field_evidence_ids, _ = _field_provenance(story, fields)
    if (
        story.story_type is not EngineeringStoryType.ARCHITECTURE_CHANGE
        or len(fields) != 2
        or not evidence_ids
        or not set(evidence_ids).intersection(field_evidence_ids)
    ):
        return None
    design_gaps = _gaps(context_gaps, _DESIGN_GAPS)
    confirmed = all(
        getattr(story, name.value).evidence_state is StoryFieldEvidenceState.CONFIRMED
        for name in fields
    )
    accepted_architecture = any(
        item.evidence_status is EvidenceStatus.ACCEPTED
        and item.evidence_type is EvidenceType.ARCHITECTURE
        for item in cluster.evidence_inputs
    )
    strength = (
        StoryOpportunitySignalStrength.STRONG
        if confirmed
        and accepted_architecture
        and cluster.lineage_state is StoryClusterLineageState.STRUCTURALLY_ANCHORED
        else StoryOpportunitySignalStrength.MODERATE
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.ARCHITECTURE_MIGRATION,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=evidence_ids,
        relevant_context_gaps=design_gaps,
        reason_code=(
            StoryOpportunityReasonCode.ARCHITECTURE_EVENT_WITH_MISSING_DECISION
            if design_gaps
            else StoryOpportunityReasonCode.ARCHITECTURE_EVENT_RECONSTRUCTED
        ),
    )


def _defensive_signal(
    story: EngineeringStory,
    cluster: StoryCluster,
    context_gaps: Sequence[StoryContextGap],
) -> StoryOpportunitySignalDecision | None:
    fields = _field_names(story, (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))
    if not {
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    }.issubset(fields):
        return None
    evidence_types = {item.evidence_type for item in _affirmative_inputs(cluster)}
    defensive_capability_ids = _capability_ids(cluster, _DEFENSIVE_CAPABILITY_TYPES)
    direct_failure = EvidenceType.FAILURE_RECOVERY in evidence_types
    bug_with_validation = (
        EvidenceType.BUG_FIX in evidence_types
        and EngineeringStoryFieldName.VALIDATION in fields
    )
    typed_story = story.story_type in {
        EngineeringStoryType.RELIABILITY_HARDENING,
        EngineeringStoryType.DEBUGGING_AND_REPAIR,
    } and bool(evidence_types & {EvidenceType.BUG_FIX, EvidenceType.FAILURE_RECOVERY})
    if not (direct_failure or defensive_capability_ids or bug_with_validation or typed_story):
        return None
    evidence_ids = _typed_evidence_ids(cluster, _HARDENING_EVIDENCE_TYPES)
    strength = (
        StoryOpportunitySignalStrength.STRONG
        if (
            direct_failure
            and EngineeringStoryFieldName.VALIDATION in fields
        )
        or len(defensive_capability_ids) >= 2
        else StoryOpportunitySignalStrength.MODERATE
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.DEFENSIVE_ENGINEERING_CLUSTER,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=defensive_capability_ids,
        relevant_context_gaps=_gaps(
            context_gaps,
            _FAILURE_GAPS | _HUMAN_GAPS,
        ),
        reason_code=StoryOpportunityReasonCode.DEFENSIVE_MECHANISM_CLUSTER,
    )


def _failure_validation_signal(
    story: EngineeringStory,
    cluster: StoryCluster,
    context_gaps: Sequence[StoryContextGap],
) -> StoryOpportunitySignalDecision | None:
    fields = _field_names(story, (
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.VALIDATION,
    ))
    evidence_types = {item.evidence_type for item in _affirmative_inputs(cluster)}
    failure_evidence_ids = set(_typed_evidence_ids(
        cluster,
        frozenset({EvidenceType.BUG_FIX, EvidenceType.FAILURE_RECOVERY}),
    ))
    problem_evidence_ids, _ = _field_provenance(
        story,
        (EngineeringStoryFieldName.PROBLEM_CONTEXT,),
    )
    if (
        len(fields) != 2
        or not evidence_types & {EvidenceType.BUG_FIX, EvidenceType.FAILURE_RECOVERY}
        or not failure_evidence_ids.intersection(problem_evidence_ids)
    ):
        return None
    evidence_ids = _typed_evidence_ids(
        cluster,
        frozenset({
            EvidenceType.BUG_FIX,
            EvidenceType.FAILURE_RECOVERY,
            EvidenceType.TESTING,
            EvidenceType.VALIDATION,
        }),
    )
    has_separate_validation_type = bool(
        evidence_types & {EvidenceType.TESTING, EvidenceType.VALIDATION}
    )
    confirmed = all(
        getattr(story, name.value).evidence_state is StoryFieldEvidenceState.CONFIRMED
        for name in fields
    )
    strength = (
        StoryOpportunitySignalStrength.STRONG
        if confirmed and has_separate_validation_type
        else StoryOpportunitySignalStrength.MODERATE
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.FAILURE_SPECIFIC_TEST_CLUSTER,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=evidence_ids,
        relevant_context_gaps=_gaps(context_gaps, _FAILURE_GAPS),
        reason_code=StoryOpportunityReasonCode.SPECIFIC_FAILURE_WITH_VALIDATION,
    )


def _repeated_subsystem_signal(
    story: EngineeringStory,
    cluster: StoryCluster,
    project_context: StoryOpportunityProjectContext | None,
    context_gaps: Sequence[StoryContextGap],
) -> StoryOpportunitySignalDecision | None:
    if project_context is None:
        return None
    current = _cluster_context(cluster)
    if current is None:
        return None
    current_keys = set(current.structural_subsystem_keys)
    if not current_keys:
        return None
    related = tuple(
        item
        for item in project_context.clusters
        if item.cluster_id != cluster.cluster_id
        and item.lineage_state is not StoryClusterLineageState.AMBIGUOUS
        and current_keys.intersection(item.structural_subsystem_keys)
    )
    if not related:
        return None
    all_types = set(current.evidence_types)
    for item in related:
        all_types.update(item.evidence_types)
    hardening_types = all_types & _HARDENING_EVIDENCE_TYPES
    if not hardening_types:
        return None
    fields = _field_names(story, (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
    ))
    if not fields:
        return None
    strength = (
        StoryOpportunitySignalStrength.STRONG
        if len(related) >= 2 and len(hardening_types) >= 2
        else StoryOpportunitySignalStrength.MODERATE
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=cluster.member_evidence_fact_ids,
        relevant_context_gaps=_gaps(
            context_gaps,
            _DESIGN_GAPS | _FAILURE_GAPS | _HUMAN_GAPS,
        ),
        related_cluster_ids=tuple(item.cluster_id for item in related),
        reason_code=StoryOpportunityReasonCode.REPEATED_SUBSYSTEM_EVENTS,
    )


def _design_decision_signal(
    story: EngineeringStory,
    cluster: StoryCluster,
    context_gaps: Sequence[StoryContextGap],
    architecture: StoryOpportunitySignalDecision | None,
) -> StoryOpportunitySignalDecision | None:
    design_gaps = _gaps(context_gaps, _DESIGN_GAPS)
    fields = _field_names(story, (
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
    ))
    evidence_types = {item.evidence_type for item in _affirmative_inputs(cluster)}
    design_evidence_ids = set(_typed_evidence_ids(cluster, _DESIGN_EVIDENCE_TYPES))
    field_evidence_ids, _ = _field_provenance(story, fields)
    if (
        not design_gaps
        or len(fields) != 2
        or story.story_type not in _DESIGN_STORY_TYPES
        or not evidence_types & _DESIGN_EVIDENCE_TYPES
        or not design_evidence_ids.intersection(field_evidence_ids)
    ):
        return None
    evidence_ids = _typed_evidence_ids(cluster, _DESIGN_EVIDENCE_TYPES)
    strength = (
        StoryOpportunitySignalStrength.STRONG
        if architecture is not None
        and architecture.strength is StoryOpportunitySignalStrength.STRONG
        and set(design_gaps) == _DESIGN_GAPS
        else StoryOpportunitySignalStrength.MODERATE
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.MAJOR_DESIGN_DECISION,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=evidence_ids,
        relevant_context_gaps=design_gaps,
        reason_code=StoryOpportunityReasonCode.SIGNIFICANT_DESIGN_CONTEXT_MISSING,
    )


def _human_context_signal(
    story: EngineeringStory,
    context_gaps: Sequence[StoryContextGap],
    event_decisions: Sequence[StoryOpportunitySignalDecision],
) -> StoryOpportunitySignalDecision | None:
    human_gaps = _gaps(context_gaps, _HUMAN_GAPS)
    if not human_gaps or not event_decisions:
        return None
    fields = _field_names(story, (
        EngineeringStoryFieldName.PROBLEM_CONTEXT,
        EngineeringStoryFieldName.MECHANISM,
        EngineeringStoryFieldName.IMPLEMENTATION,
        EngineeringStoryFieldName.VALIDATION,
        EngineeringStoryFieldName.OBSERVABLE_OUTCOME,
    ))
    evidence_ids = tuple(sorted({
        evidence_id
        for decision in event_decisions
        for evidence_id in decision.evidence_fact_ids
    }))
    capability_ids = tuple(sorted({
        capability_id
        for decision in event_decisions
        for capability_id in decision.capability_fact_ids
    }))
    strength = (
        StoryOpportunitySignalStrength.MODERATE
        if len(human_gaps) >= 2
        and any(
            decision.strength is StoryOpportunitySignalStrength.STRONG
            for decision in event_decisions
        )
        else StoryOpportunitySignalStrength.WEAK
    )
    return _signal_decision(
        signal=StoryOpportunitySignal.MISSING_HUMAN_OR_WORKFLOW_CONTEXT,
        strength=strength,
        story=story,
        supporting_fields=fields,
        evidence_fact_ids=evidence_ids,
        capability_fact_ids=capability_ids,
        relevant_context_gaps=human_gaps,
        reason_code=(
            StoryOpportunityReasonCode.MEANINGFUL_EVENT_WITH_MISSING_HUMAN_CONTEXT
        ),
    )


def _limited_strength(
    decision: StoryOpportunitySignalDecision,
    quality: StoryReconstructionQuality,
) -> StoryOpportunitySignalDecision:
    strength = decision.strength
    if quality is StoryReconstructionQuality.AMBIGUOUS:
        strength = StoryOpportunitySignalStrength.WEAK
    elif (
        quality is StoryReconstructionQuality.MINIMAL
        and strength is StoryOpportunitySignalStrength.STRONG
    ):
        strength = StoryOpportunitySignalStrength.MODERATE
    return replace(decision, strength=strength)


def _overall_level(
    decisions: Sequence[StoryOpportunitySignalDecision],
    *,
    sufficiency: EngineeringStorySufficiencyResult,
) -> StoryOpportunityLevel:
    if not decisions:
        return StoryOpportunityLevel.NONE
    quality = sufficiency.reconstruction_quality
    if quality is StoryReconstructionQuality.BLOCKED:
        return StoryOpportunityLevel.NONE
    by_signal = {decision.signal: decision for decision in decisions}
    high_value_gap = False
    architecture = by_signal.get(StoryOpportunitySignal.ARCHITECTURE_MIGRATION)
    if architecture is not None:
        high_value_gap = (
            architecture.strength is StoryOpportunitySignalStrength.STRONG
            and bool(set(architecture.relevant_context_gaps) & _DESIGN_GAPS)
        )
    failure = by_signal.get(StoryOpportunitySignal.FAILURE_SPECIFIC_TEST_CLUSTER)
    if failure is not None:
        high_value_gap = high_value_gap or (
            failure.strength is StoryOpportunitySignalStrength.STRONG
            and bool(failure.relevant_context_gaps)
        )
    repeated = by_signal.get(StoryOpportunitySignal.REPEATED_SUBSYSTEM_HARDENING)
    if repeated is not None:
        high_value_gap = high_value_gap or (
            repeated.strength is StoryOpportunitySignalStrength.STRONG
            and bool(repeated.relevant_context_gaps)
        )
    design = by_signal.get(StoryOpportunitySignal.MAJOR_DESIGN_DECISION)
    if design is not None:
        high_value_gap = high_value_gap or (
            design.strength is StoryOpportunitySignalStrength.STRONG
        )
    all_gaps = {gap for decision in decisions for gap in decision.relevant_context_gaps}
    if high_value_gap and sufficiency.claim_sufficiency.level in {
        SufficiencyLevel.MEDIUM,
        SufficiencyLevel.HIGH,
    }:
        level = StoryOpportunityLevel.HIGH
    elif all_gaps:
        level = StoryOpportunityLevel.MEDIUM
    else:
        level = StoryOpportunityLevel.LOW
    if quality is StoryReconstructionQuality.AMBIGUOUS:
        level = StoryOpportunityLevel.LOW
    elif quality is StoryReconstructionQuality.MINIMAL and level is StoryOpportunityLevel.HIGH:
        level = StoryOpportunityLevel.MEDIUM
    if sufficiency.claim_sufficiency.level is SufficiencyLevel.LOW:
        level = StoryOpportunityLevel.LOW
    return level


def _validate_inputs(
    reconstruction: StoryReconstructionResult,
    sufficiency: EngineeringStorySufficiencyResult,
    cluster: StoryCluster,
    project_context: StoryOpportunityProjectContext | None,
) -> None:
    if not isinstance(reconstruction, StoryReconstructionResult):
        raise TypeError("reconstruction_result must be a StoryReconstructionResult")
    if not isinstance(sufficiency, EngineeringStorySufficiencyResult):
        raise TypeError("sufficiency_result must be an EngineeringStorySufficiencyResult")
    if not isinstance(cluster, StoryCluster):
        raise TypeError("story_cluster must be a StoryCluster")
    project_id = reconstruction.project_id
    if sufficiency.project_id != project_id or cluster.project_id != project_id:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT
        )
    if sufficiency.cluster_id != reconstruction.cluster_id or cluster.cluster_id != reconstruction.cluster_id:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.CLUSTER_MISMATCH,
            cluster.cluster_id,
        )
    if sufficiency.reconstruction_quality is not reconstruction.reconstruction_quality:
        raise StoryOpportunityDetectionError(
            StoryOpportunityDetectionErrorCode.STORY_MISMATCH,
            reconstruction.cluster_id,
        )
    reconstructed_story = reconstruction.engineering_story
    evaluated_story = sufficiency.evaluated_story
    if reconstructed_story is None or evaluated_story is None:
        if reconstructed_story is not None or evaluated_story is not None:
            raise StoryOpportunityDetectionError(
                StoryOpportunityDetectionErrorCode.STORY_MISMATCH
            )
    else:
        expected_story = replace(
            reconstructed_story,
            claim_sufficiency=sufficiency.claim_sufficiency,
            story_sufficiency=sufficiency.story_sufficiency,
        )
        if evaluated_story != expected_story:
            raise StoryOpportunityDetectionError(
                StoryOpportunityDetectionErrorCode.STORY_MISMATCH,
                reconstructed_story.story_id,
            )
        if (
            set(cluster.member_evidence_fact_ids)
            != set(reconstructed_story.evidence_fact_ids)
            or set(cluster.member_capability_ids)
            != set(reconstructed_story.capability_fact_ids)
            or set(cluster.claim_boundary_ids)
            != set(reconstructed_story.claim_boundary_ids)
        ):
            raise StoryOpportunityDetectionError(
                StoryOpportunityDetectionErrorCode.CLUSTER_MISMATCH,
                cluster.cluster_id,
            )
    if project_context is not None:
        if not isinstance(project_context, StoryOpportunityProjectContext):
            raise TypeError("project_context must be a StoryOpportunityProjectContext")
        if project_context.project_id != project_id:
            raise StoryOpportunityDetectionError(
                StoryOpportunityDetectionErrorCode.CROSS_PROJECT_INPUT
            )


def detect_story_opportunity(
    *,
    reconstruction_result: StoryReconstructionResult,
    sufficiency_result: EngineeringStorySufficiencyResult,
    story_cluster: StoryCluster,
    project_context: StoryOpportunityProjectContext | None = None,
) -> StoryOpportunityDetectionResult:
    """Detect intrinsic story opportunity without scheduling any follow-up action."""

    _validate_inputs(
        reconstruction_result,
        sufficiency_result,
        story_cluster,
        project_context,
    )
    story = sufficiency_result.evaluated_story
    if story is None:
        return StoryOpportunityDetectionResult(
            cluster_id=reconstruction_result.cluster_id,
            project_id=reconstruction_result.project_id,
            story_id=None,
            evaluated_story=None,
            story_opportunity=StoryOpportunity(level=StoryOpportunityLevel.NONE),
            signal_decisions=(),
            relevant_context_gaps=(),
            related_project_cluster_ids=(),
            diagnostics=(StoryOpportunityDiagnosticCode.BLOCKED_RECONSTRUCTION,),
        )
    context_gaps = sufficiency_result.story_context_gaps
    architecture = _architecture_signal(story, story_cluster, context_gaps)
    defensive = _defensive_signal(story, story_cluster, context_gaps)
    failure = _failure_validation_signal(story, story_cluster, context_gaps)
    repeated = _repeated_subsystem_signal(
        story,
        story_cluster,
        project_context,
        context_gaps,
    )
    design = _design_decision_signal(
        story,
        story_cluster,
        context_gaps,
        architecture,
    )
    event_decisions = tuple(
        decision
        for decision in (architecture, defensive, failure, repeated, design)
        if decision is not None
    )
    human = _human_context_signal(story, context_gaps, event_decisions)
    decisions = tuple(
        _limited_strength(decision, reconstruction_result.reconstruction_quality)
        for decision in (*event_decisions, human)
        if decision is not None
    )
    decisions = tuple(sorted(decisions, key=lambda item: _SIGNAL_INDEX[item.signal]))
    level = _overall_level(decisions, sufficiency=sufficiency_result)
    relevant_gaps = _stable_enums(
        tuple({
            gap
            for decision in decisions
            for gap in decision.relevant_context_gaps
        }),
        StoryContextGap,
        maximum=len(StoryContextGap),
        name="relevant_context_gaps",
    )
    related_cluster_ids = _stable_cluster_ids(tuple(
        cluster_id
        for decision in decisions
        for cluster_id in decision.related_cluster_ids
    ))
    diagnostics: set[StoryOpportunityDiagnosticCode] = set()
    if not decisions:
        diagnostics.add(StoryOpportunityDiagnosticCode.NO_MEANINGFUL_EVENT_SIGNAL)
    elif not relevant_gaps:
        diagnostics.add(StoryOpportunityDiagnosticCode.NO_RELEVANT_CONTEXT_GAP)
    if reconstruction_result.reconstruction_quality is StoryReconstructionQuality.AMBIGUOUS:
        diagnostics.add(StoryOpportunityDiagnosticCode.AMBIGUOUS_RECONSTRUCTION_LIMIT)
    if reconstruction_result.reconstruction_quality is StoryReconstructionQuality.MINIMAL:
        diagnostics.add(StoryOpportunityDiagnosticCode.MINIMAL_RECONSTRUCTION_LIMIT)
    if sufficiency_result.claim_sufficiency.level is SufficiencyLevel.LOW and decisions:
        diagnostics.add(StoryOpportunityDiagnosticCode.WEAK_CLAIM_AUTHORITY_LIMIT)
    if repeated is not None:
        diagnostics.add(StoryOpportunityDiagnosticCode.REPEATED_SUBSYSTEM_CONTEXT_USED)
    opportunity = StoryOpportunity(
        level=level,
        signals=tuple(decision.signal for decision in decisions),
        missing_context=relevant_gaps,
    )
    evaluated_story = replace(story, opportunity=opportunity)
    diagnostic_index = {
        value: index for index, value in enumerate(StoryOpportunityDiagnosticCode)
    }
    return StoryOpportunityDetectionResult(
        cluster_id=reconstruction_result.cluster_id,
        project_id=reconstruction_result.project_id,
        story_id=story.story_id,
        evaluated_story=evaluated_story,
        story_opportunity=opportunity,
        signal_decisions=decisions,
        relevant_context_gaps=relevant_gaps,
        related_project_cluster_ids=related_cluster_ids,
        diagnostics=tuple(sorted(diagnostics, key=diagnostic_index.__getitem__)),
    )


__all__ = [
    "MAX_CAPABILITY_TYPES_PER_CONTEXT",
    "MAX_OPPORTUNITY_DIAGNOSTICS",
    "MAX_RELATED_PROJECT_CLUSTERS",
    "MAX_STRUCTURAL_SUBSYSTEM_KEYS",
    "StoryOpportunityClusterContext",
    "StoryOpportunityDetectionError",
    "StoryOpportunityDetectionErrorCode",
    "StoryOpportunityDetectionResult",
    "StoryOpportunityDiagnosticCode",
    "StoryOpportunityProjectContext",
    "StoryOpportunityReasonCode",
    "StoryOpportunitySignalDecision",
    "StoryOpportunitySignalStrength",
    "build_story_opportunity_project_context",
    "detect_story_opportunity",
]
