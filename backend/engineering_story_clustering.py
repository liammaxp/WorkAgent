"""Deterministic structural clustering for resolved story-evidence bundles.

The clustering policy is intentionally narrower than graph connectivity.  It
assigns evidence to one canonical event core, never closes transitively over
arbitrary relations, and leaves ambiguous or weakly anchored evidence as a
singleton.  Outputs are structural cluster candidates, not reconstructed
Engineering Stories and not persistent Story Memory.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from itertools import combinations
import re
from typing import Any

from backend.engineering_story_evidence import (
    CapabilityEvidenceLineage,
    CapabilityLineageState,
    SourceLineageState,
    StoryEventAnchor,
    StoryEventAnchorKind,
    StoryEvidenceBundle,
    StoryEvidenceInput,
    StoryEvidenceRelationStrength,
    StoryEvidenceRelationType,
)
from backend.engineering_story_models import EngineeringStoryContract
from backend.project_repository_identity import normalize_project_id


MAX_STORY_CLUSTERS = 32
MAX_EVIDENCE_MEMBERS_PER_CLUSTER = 16
MAX_MEMBERSHIP_LINKS_PER_CLUSTER = 15
MAX_CLUSTER_DECISIONS = 32
MAX_COMPETING_EVENT_CORES = 16

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_AUTHORITY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{1,299}$")
_CLUSTER_ID_RE = re.compile(r"^story_cluster_[0-9a-f]{24}$")


class StoryClusterRelationRole(str, Enum):
    ESTABLISHES_MEMBERSHIP = "establishes_membership"
    SUPPORTS_MEMBERSHIP = "supports_membership"
    CONTEXT_ONLY = "context_only"


class StoryClusterCoreKind(str, Enum):
    EXPLICIT_CHANGE = "explicit_change"
    PARENT_CHILD_CHANGE = "parent_child_change"
    COMMIT_AND_SYMBOL = "commit_and_symbol"
    SOURCE_IDENTITY = "source_identity"
    COMMIT_AND_PATH_SUPPORT = "commit_and_path_support"
    EVIDENCE_SINGLETON = "evidence_singleton"


class StoryClusterMembershipBasis(str, Enum):
    SAME_EXPLICIT_CHANGE = "same_explicit_change"
    DIRECT_PARENT_CHILD_CHANGE = "direct_parent_child_change"
    SAME_COMMIT_AND_SYMBOL = "same_commit_and_symbol"
    SAME_SOURCE_IDENTITY = "same_source_identity"
    SUPPORTED_COMMIT_AND_PATH = "supported_commit_and_path"


class StoryClusterQuality(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class StoryClusterIdentityState(str, Enum):
    STABLE_EVENT_CORE = "stable_event_core"
    CANDIDATE = "candidate"


class StoryClusterLineageState(str, Enum):
    STRUCTURALLY_ANCHORED = "structurally_anchored"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"


class StoryClusterDecisionOutcome(str, Enum):
    GROUPED = "grouped"
    SINGLETON = "singleton"
    ISOLATED_AMBIGUOUS = "isolated_ambiguous"


class StoryClusterReasonCode(str, Enum):
    SAME_EXPLICIT_CHANGE = "same_explicit_change"
    DIRECT_PARENT_CHILD_CHANGE = "direct_parent_child_change"
    SAME_COMMIT_AND_SYMBOL = "same_commit_and_symbol"
    SAME_SOURCE_IDENTITY = "same_source_identity"
    SUPPORTED_COMMIT_AND_PATH = "supported_commit_and_path"
    SINGLETON_STRONG_ANCHOR = "singleton_strong_anchor"
    SINGLETON_CONTEXT_ONLY = "singleton_context_only"
    SINGLETON_INCOMPLETE_LINEAGE = "singleton_incomplete_lineage"
    AMBIGUOUS_EVENT_CORES = "ambiguous_event_cores"


class StoryClusteringErrorCode(str, Enum):
    INVALID_BUNDLE = "invalid_bundle"
    INVALID_CLUSTER = "invalid_cluster"
    CLUSTER_BOUND_EXCEEDED = "cluster_bound_exceeded"
    CROSS_PROJECT_CLUSTER = "cross_project_cluster"


class StoryClusteringError(ValueError):
    """Bounded deterministic structural-clustering failure."""

    def __init__(
        self,
        code: StoryClusteringErrorCode | str,
        reference_id: str | None = None,
    ) -> None:
        self.code = StoryClusteringErrorCode(code)
        self.reference_id = _diagnostic_id(reference_id)
        message = self.code.value
        if self.reference_id is not None:
            message += f":{self.reference_id}"
        super().__init__(message)


_RELATION_ROLES: Mapping[
    StoryEvidenceRelationType, StoryClusterRelationRole
] = {
    StoryEvidenceRelationType.SAME_EXPLICIT_CHANGE:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryEvidenceRelationType.SAME_SOURCE:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryEvidenceRelationType.SAME_COMMIT:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
    StoryEvidenceRelationType.SAME_PATH:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
    StoryEvidenceRelationType.SAME_SYMBOL:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
    StoryEvidenceRelationType.CAPABILITY_SUPPORT_RELATION:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
    StoryEvidenceRelationType.SHARED_VALIDATED_SOURCE_LINEAGE:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
    StoryEvidenceRelationType.SAME_PARENT_CHANGE:
        StoryClusterRelationRole.CONTEXT_ONLY,
}

_CORE_INDEX = {value: index for index, value in enumerate(StoryClusterCoreKind)}
_BASIS_INDEX = {
    value: index for index, value in enumerate(StoryClusterMembershipBasis)
}
_QUALITY_INDEX = {value: index for index, value in enumerate(StoryClusterQuality)}

_CORE_ANCHOR_KINDS = {
    StoryClusterCoreKind.EXPLICIT_CHANGE: StoryEventAnchorKind.EXPLICIT_CHANGE,
    StoryClusterCoreKind.PARENT_CHILD_CHANGE: StoryEventAnchorKind.PARENT_CHANGE,
    StoryClusterCoreKind.COMMIT_AND_SYMBOL: StoryEventAnchorKind.COMMIT_AND_SYMBOL,
    StoryClusterCoreKind.SOURCE_IDENTITY: StoryEventAnchorKind.SOURCE_IDENTITY,
    StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT: StoryEventAnchorKind.COMMIT_AND_PATH,
}

_BASIS_ROLE = {
    StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryClusterMembershipBasis.DIRECT_PARENT_CHILD_CHANGE:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryClusterMembershipBasis.SAME_COMMIT_AND_SYMBOL:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryClusterMembershipBasis.SAME_SOURCE_IDENTITY:
        StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP,
    StoryClusterMembershipBasis.SUPPORTED_COMMIT_AND_PATH:
        StoryClusterRelationRole.SUPPORTS_MEMBERSHIP,
}

_BASIS_STRENGTH = {
    StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE:
        StoryEvidenceRelationStrength.STRONG,
    StoryClusterMembershipBasis.DIRECT_PARENT_CHILD_CHANGE:
        StoryEvidenceRelationStrength.STRONG,
    StoryClusterMembershipBasis.SAME_COMMIT_AND_SYMBOL:
        StoryEvidenceRelationStrength.STRONG,
    StoryClusterMembershipBasis.SAME_SOURCE_IDENTITY:
        StoryEvidenceRelationStrength.STRONG,
    StoryClusterMembershipBasis.SUPPORTED_COMMIT_AND_PATH:
        StoryEvidenceRelationStrength.MODERATE,
}


def story_cluster_relation_role(
    relation_type: StoryEvidenceRelationType | str,
) -> StoryClusterRelationRole:
    """Return policy precedence; relation strength alone never authorizes merge."""

    return _RELATION_ROLES[StoryEvidenceRelationType(relation_type)]


def _fail(
    code: StoryClusteringErrorCode,
    reference_id: str | None = None,
) -> None:
    raise StoryClusteringError(code, reference_id)


def _diagnostic_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 100
        or _CONTROL_RE.search(normalized)
        or not re.fullmatch(r"[A-Za-z0-9_.:@-]+", normalized)
    ):
        return None
    return normalized


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or value != normalized:
        _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
    return normalized


def _authority_id(value: Any, name: str, prefix: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if (
        not value.startswith(prefix)
        or not _AUTHORITY_ID_RE.fullmatch(value)
        or not value[len(prefix):]
    ):
        raise ValueError(f"{name} must be a normalized {prefix} identifier")
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
        _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
    return tuple(sorted({_authority_id(item, name, prefix) for item in values}))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class StoryClusterEventCore(EngineeringStoryContract):
    project_id: str
    core_kind: StoryClusterCoreKind
    anchor: StoryEventAnchor | None = None
    singleton_evidence_fact_id: str | None = None

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        kind = StoryClusterCoreKind(self.core_kind)
        if kind is StoryClusterCoreKind.EVIDENCE_SINGLETON:
            if self.anchor is not None or self.singleton_evidence_fact_id is None:
                raise ValueError("singleton event core requires only an evidence ID")
            singleton_id = _authority_id(
                self.singleton_evidence_fact_id,
                "singleton_evidence_fact_id",
                "pef_",
            )
        else:
            if not isinstance(self.anchor, StoryEventAnchor):
                raise TypeError("structural event core requires a StoryEventAnchor")
            if self.anchor.project_id != project_id:
                _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
            if self.anchor.anchor_kind is not _CORE_ANCHOR_KINDS[kind]:
                raise ValueError("event core kind conflicts with its structural anchor")
            if self.singleton_evidence_fact_id is not None:
                raise ValueError("structural event core cannot contain a singleton ID")
            singleton_id = None
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "core_kind", kind)
        object.__setattr__(self, "singleton_evidence_fact_id", singleton_id)


def _build_cluster_id(
    project_id: str,
    event_core: StoryClusterEventCore,
    identity_state: StoryClusterIdentityState,
    candidate_seed: str | None,
) -> str:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "event_core": event_core.to_dict(),
        "identity_state": identity_state.value,
    }
    if identity_state is StoryClusterIdentityState.CANDIDATE:
        if candidate_seed is None:
            raise ValueError("candidate cluster identity requires a stable evidence seed")
        payload["candidate_seed"] = candidate_seed
    elif candidate_seed is not None:
        raise ValueError("stable event-core identity cannot use an evidence seed")
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]
    return f"story_cluster_{digest}"


@dataclass(frozen=True, slots=True)
class StoryClusterMembershipLink(EngineeringStoryContract):
    project_id: str
    left_evidence_fact_id: str
    right_evidence_fact_id: str
    basis: StoryClusterMembershipBasis
    relation_role: StoryClusterRelationRole
    strength: StoryEvidenceRelationStrength

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        left = _authority_id(
            self.left_evidence_fact_id, "left_evidence_fact_id", "pef_"
        )
        right = _authority_id(
            self.right_evidence_fact_id, "right_evidence_fact_id", "pef_"
        )
        if left >= right:
            raise ValueError("membership link evidence IDs must be distinct and ordered")
        basis = StoryClusterMembershipBasis(self.basis)
        role = StoryClusterRelationRole(self.relation_role)
        strength = StoryEvidenceRelationStrength(self.strength)
        if role is not _BASIS_ROLE[basis] or strength is not _BASIS_STRENGTH[basis]:
            raise ValueError("membership link policy fields conflict with its basis")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "left_evidence_fact_id", left)
        object.__setattr__(self, "right_evidence_fact_id", right)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "relation_role", role)
        object.__setattr__(self, "strength", strength)


def _link_sort_key(link: StoryClusterMembershipLink) -> tuple[Any, ...]:
    return (
        link.left_evidence_fact_id,
        link.right_evidence_fact_id,
        _BASIS_INDEX[link.basis],
    )


@dataclass(frozen=True, slots=True)
class StoryCluster(EngineeringStoryContract):
    cluster_id: str
    project_id: str
    event_core: StoryClusterEventCore
    evidence_inputs: tuple[StoryEvidenceInput, ...]
    capability_lineages: tuple[CapabilityEvidenceLineage, ...]
    claim_boundary_ids: tuple[str, ...]
    membership_links: tuple[StoryClusterMembershipLink, ...]
    quality: StoryClusterQuality
    identity_state: StoryClusterIdentityState
    lineage_state: StoryClusterLineageState

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        if not isinstance(self.event_core, StoryClusterEventCore):
            raise TypeError("event_core must be a StoryClusterEventCore")
        if self.event_core.project_id != project_id:
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        if isinstance(self.evidence_inputs, (str, bytes)) or not isinstance(
            self.evidence_inputs, Sequence
        ):
            raise TypeError("evidence_inputs must be a sequence")
        if not self.evidence_inputs:
            raise ValueError("story cluster requires at least one evidence input")
        if len(self.evidence_inputs) > MAX_EVIDENCE_MEMBERS_PER_CLUSTER:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        if any(not isinstance(item, StoryEvidenceInput) for item in self.evidence_inputs):
            raise TypeError("evidence_inputs must contain StoryEvidenceInput values")
        inputs = tuple(sorted(
            self.evidence_inputs,
            key=lambda item: item.evidence_fact_id,
        ))
        if any(item.project_id != project_id for item in inputs):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        evidence_ids = tuple(item.evidence_fact_id for item in inputs)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("story cluster contains duplicate evidence inputs")
        if isinstance(self.capability_lineages, (str, bytes)) or not isinstance(
            self.capability_lineages, Sequence
        ):
            raise TypeError("capability_lineages must be a sequence")
        if any(
            not isinstance(item, CapabilityEvidenceLineage)
            for item in self.capability_lineages
        ):
            raise TypeError("invalid capability lineage in story cluster")
        capabilities = tuple(sorted(
            self.capability_lineages,
            key=lambda item: item.capability_id,
        ))
        if any(item.project_id != project_id for item in capabilities):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        capability_ids = tuple(item.capability_id for item in capabilities)
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("story cluster contains duplicate capability lineages")
        expected_capability_ids = {
            capability_id
            for item in inputs
            for capability_id in item.capability_ids
        }
        if set(capability_ids) != expected_capability_ids:
            raise ValueError("cluster capability lineages must match evidence inputs")
        boundaries = _stable_authority_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=64,
        )
        expected_boundaries = {
            boundary_id
            for item in inputs
            for boundary_id in item.claim_boundary_ids
        }
        if set(boundaries) != expected_boundaries:
            raise ValueError("cluster claim boundaries must match evidence inputs")
        if isinstance(self.membership_links, (str, bytes)) or not isinstance(
            self.membership_links, Sequence
        ):
            raise TypeError("membership_links must be a sequence")
        if len(self.membership_links) > MAX_MEMBERSHIP_LINKS_PER_CLUSTER:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        if any(
            not isinstance(item, StoryClusterMembershipLink)
            for item in self.membership_links
        ):
            raise TypeError("invalid membership link in story cluster")
        links = tuple(sorted(set(self.membership_links), key=_link_sort_key))
        valid_ids = set(evidence_ids)
        if any(
            item.project_id != project_id
            or item.left_evidence_fact_id not in valid_ids
            or item.right_evidence_fact_id not in valid_ids
            for item in links
        ):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        if len(inputs) == 1 and links:
            raise ValueError("singleton cluster cannot contain membership links")
        if len(inputs) > 1 and len(links) != len(inputs) - 1:
            raise ValueError("grouped cluster requires one bounded membership link per join")
        quality = StoryClusterQuality(self.quality)
        identity_state = StoryClusterIdentityState(self.identity_state)
        lineage_state = StoryClusterLineageState(self.lineage_state)
        if lineage_state is StoryClusterLineageState.AMBIGUOUS and len(inputs) != 1:
            raise ValueError("ambiguous evidence must remain a singleton")
        has_incomplete_lineage = any(
            lineage.state is SourceLineageState.MISSING_STRUCTURAL_CONTEXT
            for item in inputs
            for lineage in item.source_lineages
        )
        if lineage_state is not StoryClusterLineageState.AMBIGUOUS:
            expected_lineage_state = (
                StoryClusterLineageState.INCOMPLETE
                if has_incomplete_lineage
                else StoryClusterLineageState.STRUCTURALLY_ANCHORED
            )
            if lineage_state is not expected_lineage_state:
                raise ValueError("cluster lineage state conflicts with member provenance")
        expected_identity_state = _identity_state(self.event_core, lineage_state)
        if identity_state is not expected_identity_state:
            raise ValueError("cluster identity state conflicts with its event core")
        expected_quality = _quality(self.event_core, links, lineage_state)
        if quality is not expected_quality:
            raise ValueError("cluster quality conflicts with structural membership")
        candidate_seed = (
            min(evidence_ids)
            if identity_state is StoryClusterIdentityState.CANDIDATE
            else None
        )
        expected_id = _build_cluster_id(
            project_id, self.event_core, identity_state, candidate_seed
        )
        if not isinstance(self.cluster_id, str) or not _CLUSTER_ID_RE.fullmatch(
            self.cluster_id
        ) or self.cluster_id != expected_id:
            raise ValueError("cluster_id does not match canonical cluster identity")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "evidence_inputs", inputs)
        object.__setattr__(self, "capability_lineages", capabilities)
        object.__setattr__(self, "claim_boundary_ids", boundaries)
        object.__setattr__(self, "membership_links", links)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "identity_state", identity_state)
        object.__setattr__(self, "lineage_state", lineage_state)

    @property
    def member_evidence_fact_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_fact_id for item in self.evidence_inputs)

    @property
    def member_capability_ids(self) -> tuple[str, ...]:
        return tuple(item.capability_id for item in self.capability_lineages)


@dataclass(frozen=True, slots=True)
class StoryClusterDecision(EngineeringStoryContract):
    project_id: str
    evidence_fact_id: str
    cluster_id: str
    outcome: StoryClusterDecisionOutcome
    reason_code: StoryClusterReasonCode
    related_evidence_fact_ids: tuple[str, ...] = ()
    competing_core_count: int = 0

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        evidence_id = _authority_id(
            self.evidence_fact_id, "evidence_fact_id", "pef_"
        )
        if not isinstance(self.cluster_id, str) or not _CLUSTER_ID_RE.fullmatch(
            self.cluster_id
        ):
            raise ValueError("cluster_id must be a canonical story cluster identifier")
        outcome = StoryClusterDecisionOutcome(self.outcome)
        reason = StoryClusterReasonCode(self.reason_code)
        related = _stable_authority_ids(
            self.related_evidence_fact_ids,
            "related_evidence_fact_ids",
            prefix="pef_",
            maximum=MAX_EVIDENCE_MEMBERS_PER_CLUSTER - 1,
        )
        if evidence_id in related:
            raise ValueError("decision cannot relate evidence to itself")
        if isinstance(self.competing_core_count, bool) or not isinstance(
            self.competing_core_count, int
        ):
            raise TypeError("competing_core_count must be an integer")
        if not 0 <= self.competing_core_count <= MAX_COMPETING_EVENT_CORES:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        if outcome is StoryClusterDecisionOutcome.GROUPED and not related:
            raise ValueError("grouped decision requires related evidence IDs")
        if outcome is not StoryClusterDecisionOutcome.GROUPED and related:
            raise ValueError("singleton decision cannot contain related evidence IDs")
        if outcome is StoryClusterDecisionOutcome.ISOLATED_AMBIGUOUS:
            if reason is not StoryClusterReasonCode.AMBIGUOUS_EVENT_CORES:
                raise ValueError("ambiguous decision requires ambiguity reason")
            if self.competing_core_count < 2:
                raise ValueError("ambiguous decision requires competing event cores")
        elif self.competing_core_count:
            raise ValueError("only ambiguous decisions carry competing core counts")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "evidence_fact_id", evidence_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "related_evidence_fact_ids", related)


def _cluster_sort_key(cluster: StoryCluster) -> tuple[Any, ...]:
    return (
        _CORE_INDEX[cluster.event_core.core_kind],
        _canonical_json(cluster.event_core.to_dict()),
        cluster.cluster_id,
    )


@dataclass(frozen=True, slots=True)
class StoryClusteringResult(EngineeringStoryContract):
    project_id: str
    clusters: tuple[StoryCluster, ...]
    decisions: tuple[StoryClusterDecision, ...]
    capability_lineages: tuple[CapabilityEvidenceLineage, ...]
    claim_boundary_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        if isinstance(self.clusters, (str, bytes)) or not isinstance(
            self.clusters, Sequence
        ):
            raise TypeError("clusters must be a sequence")
        if not self.clusters or len(self.clusters) > MAX_STORY_CLUSTERS:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        if any(not isinstance(item, StoryCluster) for item in self.clusters):
            raise TypeError("clusters must contain StoryCluster values")
        clusters = tuple(sorted(self.clusters, key=_cluster_sort_key))
        if any(item.project_id != project_id for item in clusters):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        cluster_ids = [item.cluster_id for item in clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("clustering result contains duplicate cluster IDs")
        evidence_ids = [
            evidence_id
            for cluster in clusters
            for evidence_id in cluster.member_evidence_fact_ids
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence input cannot belong to multiple clusters")
        if isinstance(self.decisions, (str, bytes)) or not isinstance(
            self.decisions, Sequence
        ):
            raise TypeError("decisions must be a sequence")
        if len(self.decisions) > MAX_CLUSTER_DECISIONS or any(
            not isinstance(item, StoryClusterDecision) for item in self.decisions
        ):
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        decisions = tuple(sorted(
            self.decisions,
            key=lambda item: item.evidence_fact_id,
        ))
        if any(item.project_id != project_id for item in decisions):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        decision_ids = [item.evidence_fact_id for item in decisions]
        if sorted(decision_ids) != sorted(evidence_ids):
            raise ValueError("clustering decisions must cover every evidence input once")
        cluster_by_evidence = {
            evidence_id: cluster.cluster_id
            for cluster in clusters
            for evidence_id in cluster.member_evidence_fact_ids
        }
        if any(
            item.cluster_id != cluster_by_evidence[item.evidence_fact_id]
            for item in decisions
        ):
            raise ValueError("clustering decision points to the wrong cluster")
        cluster_by_id = {item.cluster_id: item for item in clusters}
        for decision in decisions:
            cluster = cluster_by_id[decision.cluster_id]
            if len(cluster.evidence_inputs) > 1:
                expected_outcome = StoryClusterDecisionOutcome.GROUPED
                expected_related = tuple(
                    item
                    for item in cluster.member_evidence_fact_ids
                    if item != decision.evidence_fact_id
                )
            elif cluster.lineage_state is StoryClusterLineageState.AMBIGUOUS:
                expected_outcome = StoryClusterDecisionOutcome.ISOLATED_AMBIGUOUS
                expected_related = ()
            else:
                expected_outcome = StoryClusterDecisionOutcome.SINGLETON
                expected_related = ()
            if (
                decision.outcome is not expected_outcome
                or decision.related_evidence_fact_ids != expected_related
            ):
                raise ValueError("clustering decision conflicts with cluster membership")
        if isinstance(self.capability_lineages, (str, bytes)) or not isinstance(
            self.capability_lineages, Sequence
        ):
            raise TypeError("capability_lineages must be a sequence")
        if any(
            not isinstance(item, CapabilityEvidenceLineage)
            for item in self.capability_lineages
        ):
            raise TypeError("invalid capability lineage in clustering result")
        capabilities = tuple(sorted(
            self.capability_lineages,
            key=lambda item: item.capability_id,
        ))
        if any(item.project_id != project_id for item in capabilities):
            _fail(StoryClusteringErrorCode.CROSS_PROJECT_CLUSTER)
        capability_ids = [item.capability_id for item in capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("clustering result contains duplicate capability lineages")
        boundaries = _stable_authority_ids(
            self.claim_boundary_ids,
            "claim_boundary_ids",
            prefix="pcb_",
            maximum=64,
        )
        if any(
            not set(cluster.member_capability_ids).issubset(capability_ids)
            or not set(cluster.claim_boundary_ids).issubset(boundaries)
            for cluster in clusters
        ):
            raise ValueError("cluster provenance is absent from the source bundle view")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "capability_lineages", capabilities)
        object.__setattr__(self, "claim_boundary_ids", boundaries)


@dataclass(frozen=True, slots=True)
class _CoreKey:
    kind: StoryClusterCoreKind
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InputView:
    evidence: StoryEvidenceInput
    explicit_changes: frozenset[str]
    parent_changes: frozenset[str]
    commit_symbols: frozenset[tuple[str, str, str]]
    commit_paths: frozenset[tuple[str, str, str]]
    source_identities: frozenset[tuple[str, str]]
    upstream_sources: frozenset[str]
    capability_ids: frozenset[str]
    lineage_incomplete: bool


def _core_key_sort(key: _CoreKey) -> tuple[Any, ...]:
    return (_CORE_INDEX[key.kind], key.values)


def _view(item: StoryEvidenceInput) -> _InputView:
    explicit = {
        anchor.explicit_change_id
        for anchor in item.event_anchors
        if anchor.anchor_kind is StoryEventAnchorKind.EXPLICIT_CHANGE
        and anchor.explicit_change_id is not None
    }
    parents = {
        anchor.parent_change_id
        for anchor in item.event_anchors
        if anchor.anchor_kind is StoryEventAnchorKind.PARENT_CHANGE
        and anchor.parent_change_id is not None
    }
    commit_symbols = {
        (anchor.repository, anchor.commit_sha, anchor.symbol)
        for anchor in item.event_anchors
        if anchor.anchor_kind is StoryEventAnchorKind.COMMIT_AND_SYMBOL
        and anchor.repository
        and anchor.commit_sha
        and anchor.symbol
    }
    commit_paths = {
        (anchor.repository, anchor.commit_sha, anchor.file_path)
        for anchor in item.event_anchors
        if anchor.anchor_kind is StoryEventAnchorKind.COMMIT_AND_PATH
        and anchor.repository
        and anchor.commit_sha
        and anchor.file_path
    }
    source_identities = {
        (anchor.source_type, anchor.source_id)
        for anchor in item.event_anchors
        if anchor.anchor_kind is StoryEventAnchorKind.SOURCE_IDENTITY
        and anchor.source_type
        and anchor.source_id
    }
    upstream = {
        lineage.upstream_source_id
        for lineage in item.source_lineages
        if lineage.upstream_source_id
    }
    return _InputView(
        evidence=item,
        explicit_changes=frozenset(explicit),
        parent_changes=frozenset(parents),
        commit_symbols=frozenset(commit_symbols),
        commit_paths=frozenset(commit_paths),
        source_identities=frozenset(source_identities),
        upstream_sources=frozenset(upstream),
        capability_ids=frozenset(item.capability_ids),
        lineage_incomplete=any(
            lineage.state is SourceLineageState.MISSING_STRUCTURAL_CONTEXT
            for lineage in item.source_lineages
        ),
    )


def _conflicting_explicit_events(left: _InputView, right: _InputView) -> bool:
    if not left.explicit_changes or not right.explicit_changes:
        return False
    if left.explicit_changes & right.explicit_changes:
        return False
    return not bool(
        (left.explicit_changes & right.parent_changes)
        or (right.explicit_changes & left.parent_changes)
    )


def _conflicting_commit_symbols(left: _InputView, right: _InputView) -> bool:
    return bool(
        left.commit_symbols
        and right.commit_symbols
        and not (left.commit_symbols & right.commit_symbols)
    )


def _support_pair(left: _InputView, right: _InputView, key: _CoreKey) -> bool:
    if key.kind is not StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT:
        return False
    path = tuple(key.values)
    if path not in left.commit_paths or path not in right.commit_paths:
        return False
    if _conflicting_explicit_events(left, right) or _conflicting_commit_symbols(
        left, right
    ):
        return False
    return bool(
        (left.upstream_sources & right.upstream_sources)
        or (left.capability_ids & right.capability_ids)
    )


def _candidate_groups(
    views: Mapping[str, _InputView],
) -> dict[_CoreKey, frozenset[str]]:
    groups: dict[_CoreKey, set[str]] = defaultdict(set)
    for evidence_id, view in views.items():
        for value in view.explicit_changes:
            groups[_CoreKey(StoryClusterCoreKind.EXPLICIT_CHANGE, (value,))].add(
                evidence_id
            )
        for value in view.commit_symbols:
            groups[_CoreKey(StoryClusterCoreKind.COMMIT_AND_SYMBOL, value)].add(
                evidence_id
            )
        for value in view.source_identities:
            groups[_CoreKey(StoryClusterCoreKind.SOURCE_IDENTITY, value)].add(
                evidence_id
            )
    for change_id in sorted({
        value for view in views.values() for value in view.explicit_changes
    }):
        parents = {
            evidence_id
            for evidence_id, view in views.items()
            if change_id in view.explicit_changes
        }
        children = {
            evidence_id
            for evidence_id, view in views.items()
            if change_id in view.parent_changes
        }
        if parents and children:
            groups.pop(
                _CoreKey(StoryClusterCoreKind.EXPLICIT_CHANGE, (change_id,)),
                None,
            )
            groups[_CoreKey(
                StoryClusterCoreKind.PARENT_CHILD_CHANGE,
                (change_id,),
            )].update(parents | children)
    for path in sorted({value for view in views.values() for value in view.commit_paths}):
        key = _CoreKey(StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT, path)
        members = [
            evidence_id
            for evidence_id, view in views.items()
            if path in view.commit_paths
        ]
        if len(members) >= 2 and all(
            _support_pair(views[left], views[right], key)
            for left, right in combinations(sorted(members), 2)
        ):
            groups[key].update(members)
    eligible: dict[_CoreKey, frozenset[str]] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        if key.kind is StoryClusterCoreKind.COMMIT_AND_SYMBOL:
            pairs = combinations(sorted(members), 2)
            if any(
                _conflicting_explicit_events(views[left], views[right])
                for left, right in pairs
            ):
                continue
        if len(members) > MAX_EVIDENCE_MEMBERS_PER_CLUSTER:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
        eligible[key] = frozenset(members)
    return dict(sorted(eligible.items(), key=lambda item: _core_key_sort(item[0])))


def _anchor_from_key(project_id: str, key: _CoreKey) -> StoryEventAnchor:
    if key.kind is StoryClusterCoreKind.EXPLICIT_CHANGE:
        return StoryEventAnchor(
            project_id=project_id,
            anchor_kind=StoryEventAnchorKind.EXPLICIT_CHANGE,
            strength=StoryEvidenceRelationStrength.STRONG,
            explicit_change_id=key.values[0],
        )
    if key.kind is StoryClusterCoreKind.PARENT_CHILD_CHANGE:
        return StoryEventAnchor(
            project_id=project_id,
            anchor_kind=StoryEventAnchorKind.PARENT_CHANGE,
            strength=StoryEvidenceRelationStrength.MODERATE,
            parent_change_id=key.values[0],
        )
    if key.kind is StoryClusterCoreKind.COMMIT_AND_SYMBOL:
        return StoryEventAnchor(
            project_id=project_id,
            anchor_kind=StoryEventAnchorKind.COMMIT_AND_SYMBOL,
            strength=StoryEvidenceRelationStrength.STRONG,
            repository=key.values[0],
            commit_sha=key.values[1],
            symbol=key.values[2],
        )
    if key.kind is StoryClusterCoreKind.SOURCE_IDENTITY:
        return StoryEventAnchor(
            project_id=project_id,
            anchor_kind=StoryEventAnchorKind.SOURCE_IDENTITY,
            strength=StoryEvidenceRelationStrength.WEAK,
            source_type=key.values[0],
            source_id=key.values[1],
        )
    if key.kind is StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT:
        return StoryEventAnchor(
            project_id=project_id,
            anchor_kind=StoryEventAnchorKind.COMMIT_AND_PATH,
            strength=StoryEvidenceRelationStrength.MODERATE,
            repository=key.values[0],
            commit_sha=key.values[1],
            file_path=key.values[2],
        )
    raise ValueError("singleton evidence core has no structural anchor")


def _event_core(
    project_id: str,
    group_key: _CoreKey | None,
    members: Sequence[_InputView],
    *,
    ambiguous: bool,
) -> StoryClusterEventCore:
    if ambiguous:
        return StoryClusterEventCore(
            project_id=project_id,
            core_kind=StoryClusterCoreKind.EVIDENCE_SINGLETON,
            singleton_evidence_fact_id=members[0].evidence.evidence_fact_id,
        )
    explicit = sorted({
        value for member in members for value in member.explicit_changes
    })
    if len(explicit) == 1 and (
        group_key is None
        or group_key.kind is not StoryClusterCoreKind.PARENT_CHILD_CHANGE
    ):
        key = _CoreKey(StoryClusterCoreKind.EXPLICIT_CHANGE, (explicit[0],))
        return StoryClusterEventCore(
            project_id=project_id,
            core_kind=key.kind,
            anchor=_anchor_from_key(project_id, key),
        )
    if group_key is not None:
        return StoryClusterEventCore(
            project_id=project_id,
            core_kind=group_key.kind,
            anchor=_anchor_from_key(project_id, group_key),
        )
    member = members[0]
    single_keys: list[_CoreKey] = []
    single_keys.extend(
        _CoreKey(StoryClusterCoreKind.EXPLICIT_CHANGE, (value,))
        for value in member.explicit_changes
    )
    single_keys.extend(
        _CoreKey(StoryClusterCoreKind.COMMIT_AND_SYMBOL, value)
        for value in member.commit_symbols
    )
    single_keys.extend(
        _CoreKey(StoryClusterCoreKind.SOURCE_IDENTITY, value)
        for value in member.source_identities
    )
    single_keys.extend(
        _CoreKey(StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT, value)
        for value in member.commit_paths
    )
    single_keys = sorted(set(single_keys), key=_core_key_sort)
    best_rank = _CORE_INDEX[single_keys[0].kind] if single_keys else None
    best = [
        key for key in single_keys if _CORE_INDEX[key.kind] == best_rank
    ] if best_rank is not None else []
    if len(best) == 1:
        return StoryClusterEventCore(
            project_id=project_id,
            core_kind=best[0].kind,
            anchor=_anchor_from_key(project_id, best[0]),
        )
    return StoryClusterEventCore(
        project_id=project_id,
        core_kind=StoryClusterCoreKind.EVIDENCE_SINGLETON,
        singleton_evidence_fact_id=member.evidence.evidence_fact_id,
    )


def _basis_for_core(kind: StoryClusterCoreKind) -> StoryClusterMembershipBasis:
    return {
        StoryClusterCoreKind.EXPLICIT_CHANGE:
            StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE,
        StoryClusterCoreKind.PARENT_CHILD_CHANGE:
            StoryClusterMembershipBasis.DIRECT_PARENT_CHILD_CHANGE,
        StoryClusterCoreKind.COMMIT_AND_SYMBOL:
            StoryClusterMembershipBasis.SAME_COMMIT_AND_SYMBOL,
        StoryClusterCoreKind.SOURCE_IDENTITY:
            StoryClusterMembershipBasis.SAME_SOURCE_IDENTITY,
        StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT:
            StoryClusterMembershipBasis.SUPPORTED_COMMIT_AND_PATH,
    }[kind]


def _membership_links(
    project_id: str,
    group_key: _CoreKey | None,
    members: Sequence[_InputView],
) -> tuple[StoryClusterMembershipLink, ...]:
    if group_key is None or len(members) < 2:
        return ()
    basis = _basis_for_core(group_key.kind)
    ordered = sorted(members, key=lambda item: item.evidence.evidence_fact_id)
    if group_key.kind is StoryClusterCoreKind.PARENT_CHILD_CHANGE:
        parent_id = group_key.values[0]
        roots = [item for item in ordered if parent_id in item.explicit_changes]
        root = roots[0]
    else:
        root = ordered[0]
    links: list[StoryClusterMembershipLink] = []
    for member in ordered:
        if member.evidence.evidence_fact_id == root.evidence.evidence_fact_id:
            continue
        member_basis = basis
        if (
            group_key.kind is StoryClusterCoreKind.PARENT_CHILD_CHANGE
            and group_key.values[0] in member.explicit_changes
        ):
            member_basis = StoryClusterMembershipBasis.SAME_EXPLICIT_CHANGE
        left, right = sorted((
            root.evidence.evidence_fact_id,
            member.evidence.evidence_fact_id,
        ))
        links.append(StoryClusterMembershipLink(
            project_id=project_id,
            left_evidence_fact_id=left,
            right_evidence_fact_id=right,
            basis=member_basis,
            relation_role=_BASIS_ROLE[member_basis],
            strength=_BASIS_STRENGTH[member_basis],
        ))
    return tuple(sorted(links, key=_link_sort_key))


def _lineage_state(
    members: Sequence[_InputView],
    *,
    ambiguous: bool,
) -> StoryClusterLineageState:
    if ambiguous:
        return StoryClusterLineageState.AMBIGUOUS
    if any(item.lineage_incomplete for item in members):
        return StoryClusterLineageState.INCOMPLETE
    return StoryClusterLineageState.STRUCTURALLY_ANCHORED


def _identity_state(
    core: StoryClusterEventCore,
    lineage_state: StoryClusterLineageState,
) -> StoryClusterIdentityState:
    if (
        lineage_state is not StoryClusterLineageState.AMBIGUOUS
        and core.core_kind
        in {
            StoryClusterCoreKind.EXPLICIT_CHANGE,
            StoryClusterCoreKind.PARENT_CHILD_CHANGE,
            StoryClusterCoreKind.COMMIT_AND_SYMBOL,
            StoryClusterCoreKind.SOURCE_IDENTITY,
        }
        and not (
            core.core_kind is StoryClusterCoreKind.SOURCE_IDENTITY
            and lineage_state is StoryClusterLineageState.INCOMPLETE
        )
    ):
        return StoryClusterIdentityState.STABLE_EVENT_CORE
    return StoryClusterIdentityState.CANDIDATE


def _quality(
    core: StoryClusterEventCore,
    links: Sequence[StoryClusterMembershipLink],
    lineage_state: StoryClusterLineageState,
) -> StoryClusterQuality:
    if lineage_state in {
        StoryClusterLineageState.AMBIGUOUS,
        StoryClusterLineageState.INCOMPLETE,
    }:
        return StoryClusterQuality.WEAK
    if links and all(
        item.relation_role is StoryClusterRelationRole.ESTABLISHES_MEMBERSHIP
        for item in links
    ):
        return StoryClusterQuality.STRONG
    if links:
        return StoryClusterQuality.MODERATE
    if core.core_kind is StoryClusterCoreKind.EXPLICIT_CHANGE:
        return StoryClusterQuality.STRONG
    if core.core_kind in {
        StoryClusterCoreKind.COMMIT_AND_SYMBOL,
        StoryClusterCoreKind.SOURCE_IDENTITY,
        StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT,
    }:
        return StoryClusterQuality.MODERATE
    return StoryClusterQuality.WEAK


def _reason_for_group(key: _CoreKey) -> StoryClusterReasonCode:
    return {
        StoryClusterCoreKind.EXPLICIT_CHANGE:
            StoryClusterReasonCode.SAME_EXPLICIT_CHANGE,
        StoryClusterCoreKind.PARENT_CHILD_CHANGE:
            StoryClusterReasonCode.DIRECT_PARENT_CHILD_CHANGE,
        StoryClusterCoreKind.COMMIT_AND_SYMBOL:
            StoryClusterReasonCode.SAME_COMMIT_AND_SYMBOL,
        StoryClusterCoreKind.SOURCE_IDENTITY:
            StoryClusterReasonCode.SAME_SOURCE_IDENTITY,
        StoryClusterCoreKind.COMMIT_AND_PATH_SUPPORT:
            StoryClusterReasonCode.SUPPORTED_COMMIT_AND_PATH,
    }[key.kind]


def _build_cluster(
    bundle: StoryEvidenceBundle,
    views: Mapping[str, _InputView],
    member_ids: Sequence[str],
    group_key: _CoreKey | None,
    *,
    ambiguous: bool,
) -> StoryCluster:
    members = tuple(views[evidence_id] for evidence_id in sorted(member_ids))
    if len(members) > MAX_EVIDENCE_MEMBERS_PER_CLUSTER:
        _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED)
    core = _event_core(bundle.project_id, group_key, members, ambiguous=ambiguous)
    links = _membership_links(bundle.project_id, group_key, members)
    lineage_state = _lineage_state(members, ambiguous=ambiguous)
    identity_state = _identity_state(core, lineage_state)
    evidence_inputs = tuple(item.evidence for item in members)
    member_capability_ids = {
        capability_id
        for item in evidence_inputs
        for capability_id in item.capability_ids
    }
    capabilities = tuple(
        item
        for item in bundle.capability_lineages
        if item.capability_id in member_capability_ids
    )
    boundaries = tuple(sorted({
        boundary_id
        for item in evidence_inputs
        for boundary_id in item.claim_boundary_ids
    }))
    candidate_seed = (
        min(member_ids)
        if identity_state is StoryClusterIdentityState.CANDIDATE
        else None
    )
    cluster_id = _build_cluster_id(
        bundle.project_id, core, identity_state, candidate_seed
    )
    return StoryCluster(
        cluster_id=cluster_id,
        project_id=bundle.project_id,
        event_core=core,
        evidence_inputs=evidence_inputs,
        capability_lineages=capabilities,
        claim_boundary_ids=boundaries,
        membership_links=links,
        quality=_quality(core, links, lineage_state),
        identity_state=identity_state,
        lineage_state=lineage_state,
    )


def cluster_story_evidence_bundle(
    bundle: StoryEvidenceBundle,
) -> StoryClusteringResult:
    """Cluster one validated bundle without I/O, semantic inference, or traversal."""

    if not isinstance(bundle, StoryEvidenceBundle):
        raise TypeError("bundle must be a StoryEvidenceBundle")
    project_id = _exact_project_id(bundle.project_id)
    views = {
        item.evidence_fact_id: _view(item)
        for item in bundle.evidence_inputs
    }
    if len(views) != len(bundle.evidence_inputs):
        _fail(StoryClusteringErrorCode.INVALID_BUNDLE)
    groups = _candidate_groups(views)
    candidate_keys: dict[str, list[_CoreKey]] = defaultdict(list)
    for key, evidence_ids in groups.items():
        for evidence_id in evidence_ids:
            candidate_keys[evidence_id].append(key)
    selected_key: dict[str, _CoreKey] = {}
    ambiguous_counts: dict[str, int] = {}
    for evidence_id in sorted(views):
        keys = sorted(set(candidate_keys.get(evidence_id, ())), key=_core_key_sort)
        if not keys:
            continue
        best_rank = _CORE_INDEX[keys[0].kind]
        best = [key for key in keys if _CORE_INDEX[key.kind] == best_rank]
        if len(best) > MAX_COMPETING_EVENT_CORES:
            _fail(StoryClusteringErrorCode.CLUSTER_BOUND_EXCEEDED, evidence_id)
        if len(best) == 1:
            selected_key[evidence_id] = best[0]
        else:
            ambiguous_counts[evidence_id] = len(best)
    assigned_groups: dict[_CoreKey, list[str]] = defaultdict(list)
    for evidence_id, key in selected_key.items():
        assigned_groups[key].append(evidence_id)
    clusters: list[StoryCluster] = []
    decision_specs: dict[
        str,
        tuple[
            StoryClusterDecisionOutcome,
            StoryClusterReasonCode,
            tuple[str, ...],
            int,
        ],
    ] = {}
    assigned: set[str] = set()
    for key in sorted(assigned_groups, key=_core_key_sort):
        member_ids = tuple(sorted(assigned_groups[key]))
        if len(member_ids) < 2:
            continue
        cluster = _build_cluster(
            bundle, views, member_ids, key, ambiguous=False
        )
        clusters.append(cluster)
        assigned.update(member_ids)
        reason = _reason_for_group(key)
        for evidence_id in member_ids:
            decision_specs[evidence_id] = (
                StoryClusterDecisionOutcome.GROUPED,
                reason,
                tuple(item for item in member_ids if item != evidence_id),
                0,
            )
    for evidence_id in sorted(views):
        if evidence_id in assigned:
            continue
        ambiguous = evidence_id in ambiguous_counts
        cluster = _build_cluster(
            bundle,
            views,
            (evidence_id,),
            None,
            ambiguous=ambiguous,
        )
        clusters.append(cluster)
        view = views[evidence_id]
        if ambiguous:
            spec = (
                StoryClusterDecisionOutcome.ISOLATED_AMBIGUOUS,
                StoryClusterReasonCode.AMBIGUOUS_EVENT_CORES,
                (),
                ambiguous_counts[evidence_id],
            )
        elif view.lineage_incomplete:
            spec = (
                StoryClusterDecisionOutcome.SINGLETON,
                StoryClusterReasonCode.SINGLETON_INCOMPLETE_LINEAGE,
                (),
                0,
            )
        elif cluster.event_core.core_kind in {
            StoryClusterCoreKind.EXPLICIT_CHANGE,
            StoryClusterCoreKind.COMMIT_AND_SYMBOL,
            StoryClusterCoreKind.SOURCE_IDENTITY,
        }:
            spec = (
                StoryClusterDecisionOutcome.SINGLETON,
                StoryClusterReasonCode.SINGLETON_STRONG_ANCHOR,
                (),
                0,
            )
        else:
            spec = (
                StoryClusterDecisionOutcome.SINGLETON,
                StoryClusterReasonCode.SINGLETON_CONTEXT_ONLY,
                (),
                0,
            )
        decision_specs[evidence_id] = spec
    ordered_clusters = tuple(sorted(clusters, key=_cluster_sort_key))
    cluster_by_evidence = {
        evidence_id: cluster.cluster_id
        for cluster in ordered_clusters
        for evidence_id in cluster.member_evidence_fact_ids
    }
    decisions = tuple(
        StoryClusterDecision(
            project_id=project_id,
            evidence_fact_id=evidence_id,
            cluster_id=cluster_by_evidence[evidence_id],
            outcome=decision_specs[evidence_id][0],
            reason_code=decision_specs[evidence_id][1],
            related_evidence_fact_ids=decision_specs[evidence_id][2],
            competing_core_count=decision_specs[evidence_id][3],
        )
        for evidence_id in sorted(views)
    )
    return StoryClusteringResult(
        project_id=project_id,
        clusters=ordered_clusters,
        decisions=decisions,
        capability_lineages=tuple(bundle.capability_lineages),
        claim_boundary_ids=tuple(bundle.claim_boundary_ids),
    )


__all__ = [
    "MAX_CLUSTER_DECISIONS",
    "MAX_COMPETING_EVENT_CORES",
    "MAX_EVIDENCE_MEMBERS_PER_CLUSTER",
    "MAX_MEMBERSHIP_LINKS_PER_CLUSTER",
    "MAX_STORY_CLUSTERS",
    "StoryCluster",
    "StoryClusterCoreKind",
    "StoryClusterDecision",
    "StoryClusterDecisionOutcome",
    "StoryClusterEventCore",
    "StoryClusterIdentityState",
    "StoryClusterLineageState",
    "StoryClusterMembershipBasis",
    "StoryClusterMembershipLink",
    "StoryClusterQuality",
    "StoryClusterReasonCode",
    "StoryClusterRelationRole",
    "StoryClusteringError",
    "StoryClusteringErrorCode",
    "StoryClusteringResult",
    "cluster_story_evidence_bundle",
    "story_cluster_relation_role",
]
