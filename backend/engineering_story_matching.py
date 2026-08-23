"""Deterministic project-scoped identity resolution for engineering stories.

The matcher consumes only validated Story Memory records.  It establishes
canonical identity from durable structural provenance, never from prose,
quality, sufficiency, opportunity, job context, or presentation state.  It
does not rewrite story content or perform lifecycle/version transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any

from backend.engineering_story_clustering import StoryClusterIdentityState
from backend.engineering_story_memory import (
    MAX_ENGINEERING_STORY_MEMORY_RECORDS,
    EngineeringStoryMemory,
    EngineeringStoryMemoryRecord,
)
from backend.engineering_story_models import (
    EngineeringStoryContract,
    validate_engineering_story_id,
)
from backend.project_repository_identity import normalize_project_id


MAX_CANONICALIZATION_CANDIDATES = MAX_ENGINEERING_STORY_MEMORY_RECORDS
MAX_CANONICAL_STORY_IDENTITIES = 2_048
MAX_CANONICAL_STORY_ALIASES = 1_024
MAX_CANONICAL_COMPARISONS = 262_144
MAX_ALIAS_DEPTH = 32
MAX_MERGE_PARTICIPANTS = 32
MAX_SPLIT_CHILDREN = 32
MAX_CANONICALIZATION_DECISIONS = 2_048
MAX_CANONICALIZATION_DIAGNOSTICS = 32

_CANONICAL_ID_RE = re.compile(r"^engineering_story_[0-9a-f]{24}$")
_CANDIDATE_ID_RE = re.compile(r"^engineering_story_candidate_[0-9a-f]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanonicalStorySeedKind(str, Enum):
    STABLE_EVENT_CORE = "stable_event_core"
    PROVISIONAL_FOUNDING_IDENTITY = "provisional_founding_identity"


class StoryMatchSignalCategory(str, Enum):
    IDENTITY_ESTABLISHING = "identity_establishing"
    IDENTITY_SUPPORTING = "identity_supporting"
    CONTEXT_ONLY = "context_only"


class StoryMatchSignal(str, Enum):
    EXACT_CANDIDATE_IDENTITY = "exact_candidate_identity"
    SAME_DURABLE_EVENT_CORE = "same_durable_event_core"
    EVIDENCE_AUTHORITY_CONTINUITY = "evidence_authority_continuity"
    SOURCE_LINEAGE_CONTINUITY = "source_lineage_continuity"
    SAME_STORY_TYPE = "same_story_type"
    CAPABILITY_CONTEXT_OVERLAP = "capability_context_overlap"


_SIGNAL_CATEGORIES = {
    StoryMatchSignal.EXACT_CANDIDATE_IDENTITY:
        StoryMatchSignalCategory.IDENTITY_ESTABLISHING,
    StoryMatchSignal.SAME_DURABLE_EVENT_CORE:
        StoryMatchSignalCategory.IDENTITY_ESTABLISHING,
    StoryMatchSignal.EVIDENCE_AUTHORITY_CONTINUITY:
        StoryMatchSignalCategory.IDENTITY_SUPPORTING,
    StoryMatchSignal.SOURCE_LINEAGE_CONTINUITY:
        StoryMatchSignalCategory.IDENTITY_SUPPORTING,
    StoryMatchSignal.SAME_STORY_TYPE: StoryMatchSignalCategory.CONTEXT_ONLY,
    StoryMatchSignal.CAPABILITY_CONTEXT_OVERLAP:
        StoryMatchSignalCategory.CONTEXT_ONLY,
}


def classify_story_match_signal(
    signal: StoryMatchSignal,
) -> StoryMatchSignalCategory:
    return _SIGNAL_CATEGORIES[StoryMatchSignal(signal)]


class StoryMatchOutcome(str, Enum):
    EXACT_MATCH = "exact_match"
    STRONG_MATCH = "strong_match"
    NEW_CANONICAL = "new_canonical"
    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"
    MERGE_CANDIDATE = "merge_candidate"
    SPLIT_CANDIDATE = "split_candidate"


class StoryMatchReasonCode(str, Enum):
    EXACT_CANDIDATE_MATCH = "exact_candidate_match"
    SAME_DURABLE_EVENT_CORE = "same_durable_event_core"
    STRONG_PROVENANCE_CONTINUITY = "strong_provenance_continuity"
    NEW_DISTINCT_EVENT = "new_distinct_event"
    WEAK_CONTEXT_ONLY = "weak_context_only"
    AMBIGUOUS_MULTI_MATCH = "ambiguous_multi_match"
    MERGE_STRONG_COMMON_CORE = "merge_strong_common_core"
    SPLIT_DISTINCT_EVENT_CORES = "split_distinct_event_cores"
    SPLIT_AMBIGUOUS_PARENT = "split_ambiguous_parent"


class StoryMergeOutcome(str, Enum):
    MERGED = "merged"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    AMBIGUOUS = "ambiguous"


class StorySplitOutcome(str, Enum):
    SPLIT = "split"
    AMBIGUOUS = "ambiguous"


class StoryCanonicalizationDiagnosticCode(str, Enum):
    CROSS_PROJECT_CANDIDATE_ISOLATED = "cross_project_candidate_isolated"
    CONTEXT_ONLY_MATCH_REJECTED = "context_only_match_rejected"
    AMBIGUOUS_MATCH_PRESERVED = "ambiguous_match_preserved"
    AMBIGUOUS_SPLIT_PRESERVED = "ambiguous_split_preserved"


class EngineeringStoryMatchingErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BOUND_EXCEEDED = "bound_exceeded"
    CONFLICTING_IDENTITY = "conflicting_identity"
    CONFLICTING_CANDIDATE_LINK = "conflicting_candidate_link"
    CONFLICTING_ALIAS = "conflicting_alias"
    ALIAS_CYCLE = "alias_cycle"
    ALIAS_DEPTH_EXCEEDED = "alias_depth_exceeded"
    CROSS_PROJECT_ALIAS = "cross_project_alias"
    INVALID_SPLIT_RELATIONSHIP = "invalid_split_relationship"
    COMPARISON_BOUND_EXCEEDED = "comparison_bound_exceeded"


class EngineeringStoryMatchingError(ValueError):
    """Bounded fail-closed canonicalization error."""

    def __init__(self, code: EngineeringStoryMatchingErrorCode | str) -> None:
        self.code = EngineeringStoryMatchingErrorCode(code)
        super().__init__(self.code.value)


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
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.INVALID_INPUT
        )
    return normalized


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.INVALID_INPUT
        )
    return value


def _canonical_story_id(value: Any) -> str:
    if not isinstance(value, str) or not _CANONICAL_ID_RE.fullmatch(value):
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.INVALID_INPUT
        )
    return value


def _candidate_story_id(value: Any) -> str:
    candidate_id = validate_engineering_story_id(value)
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.INVALID_INPUT
        )
    return candidate_id


def _stable_ids(
    values: Sequence[str],
    validator: Any,
    *,
    maximum: int,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
        )
    return tuple(sorted({validator(value) for value in values}))


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
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
        )
    index = {value: position for position, value in enumerate(enum_type)}
    normalized = {enum_type(value) for value in values}
    return tuple(sorted(normalized, key=index.__getitem__))


def build_canonical_engineering_story_id(
    *,
    project_id: str,
    seed_kind: CanonicalStorySeedKind,
    seed_fingerprint: str,
) -> str:
    """Build a canonical ID from one immutable founding structural seed."""

    project = _exact_project_id(project_id)
    kind = CanonicalStorySeedKind(seed_kind)
    fingerprint = _digest(seed_fingerprint)
    digest = _fingerprint({
        "project_id": project,
        "seed_kind": kind.value,
        "seed_fingerprint": fingerprint,
    })[:24]
    return f"engineering_story_{digest}"


@dataclass(frozen=True, slots=True)
class CanonicalEngineeringStoryIdentity(EngineeringStoryContract):
    project_id: str
    canonical_story_id: str
    founding_seed_kind: CanonicalStorySeedKind
    founding_seed_fingerprint: str

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        seed_kind = CanonicalStorySeedKind(self.founding_seed_kind)
        seed = _digest(self.founding_seed_fingerprint)
        canonical_id = _canonical_story_id(self.canonical_story_id)
        expected = build_canonical_engineering_story_id(
            project_id=project_id,
            seed_kind=seed_kind,
            seed_fingerprint=seed,
        )
        if canonical_id != expected:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.CONFLICTING_IDENTITY
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "founding_seed_kind", seed_kind)
        object.__setattr__(self, "founding_seed_fingerprint", seed)


@dataclass(frozen=True, slots=True)
class CandidateCanonicalStoryLink(EngineeringStoryContract):
    project_id: str
    candidate_story_id: str
    canonical_story_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _exact_project_id(self.project_id))
        object.__setattr__(
            self,
            "candidate_story_id",
            _candidate_story_id(self.candidate_story_id),
        )
        object.__setattr__(
            self,
            "canonical_story_id",
            _canonical_story_id(self.canonical_story_id),
        )


@dataclass(frozen=True, slots=True)
class StoryIdentityAlias(EngineeringStoryContract):
    project_id: str
    alias_story_id: str
    canonical_story_id: str

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        alias_id = _canonical_story_id(self.alias_story_id)
        target_id = _canonical_story_id(self.canonical_story_id)
        if alias_id == target_id:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.CONFLICTING_ALIAS
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "alias_story_id", alias_id)
        object.__setattr__(self, "canonical_story_id", target_id)


@dataclass(frozen=True, slots=True)
class StorySplitRelationship(EngineeringStoryContract):
    project_id: str
    parent_canonical_story_id: str
    child_candidate_story_ids: tuple[str, ...]
    child_canonical_story_ids: tuple[str, ...]
    retained_parent_candidate_story_id: str | None
    outcome: StorySplitOutcome

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        parent_id = _canonical_story_id(self.parent_canonical_story_id)
        candidates = _stable_ids(
            self.child_candidate_story_ids,
            _candidate_story_id,
            maximum=MAX_SPLIT_CHILDREN,
            name="child_candidate_story_ids",
        )
        children = _stable_ids(
            self.child_canonical_story_ids,
            _canonical_story_id,
            maximum=MAX_SPLIT_CHILDREN,
            name="child_canonical_story_ids",
        )
        retained = (
            None
            if self.retained_parent_candidate_story_id is None
            else _candidate_story_id(self.retained_parent_candidate_story_id)
        )
        outcome = StorySplitOutcome(self.outcome)
        if len(candidates) < 2:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
            )
        if outcome is StorySplitOutcome.SPLIT:
            if (
                retained not in candidates
                or parent_id not in children
                or len(children) != len(candidates)
            ):
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
                )
        elif children or retained is not None:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "parent_canonical_story_id", parent_id)
        object.__setattr__(self, "child_candidate_story_ids", candidates)
        object.__setattr__(self, "child_canonical_story_ids", children)
        object.__setattr__(self, "retained_parent_candidate_story_id", retained)
        object.__setattr__(self, "outcome", outcome)


def _resolve_alias_target(
    story_id: str,
    aliases: Mapping[str, StoryIdentityAlias],
) -> str:
    current = story_id
    visited: set[str] = set()
    for _ in range(MAX_ALIAS_DEPTH + 1):
        item = aliases.get(current)
        if item is None:
            return current
        if current in visited:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.ALIAS_CYCLE
            )
        visited.add(current)
        current = item.canonical_story_id
    raise EngineeringStoryMatchingError(
        EngineeringStoryMatchingErrorCode.ALIAS_DEPTH_EXCEEDED
    )


@dataclass(frozen=True, slots=True)
class EngineeringStoryIdentityMap(EngineeringStoryContract):
    canonical_identities: tuple[CanonicalEngineeringStoryIdentity, ...] = ()
    candidate_links: tuple[CandidateCanonicalStoryLink, ...] = ()
    aliases: tuple[StoryIdentityAlias, ...] = ()
    split_relationships: tuple[StorySplitRelationship, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.canonical_identities, (str, bytes))
            or not isinstance(self.canonical_identities, Sequence)
            or any(
                not isinstance(item, CanonicalEngineeringStoryIdentity)
                for item in self.canonical_identities
            )
        ):
            raise TypeError("canonical_identities must contain canonical identities")
        if len(self.canonical_identities) > MAX_CANONICAL_STORY_IDENTITIES:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
            )
        identities_by_id: dict[str, CanonicalEngineeringStoryIdentity] = {}
        for identity in self.canonical_identities:
            previous = identities_by_id.get(identity.canonical_story_id)
            if previous is not None and previous != identity:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CONFLICTING_IDENTITY
                )
            identities_by_id[identity.canonical_story_id] = identity
        identities = tuple(sorted(
            identities_by_id.values(),
            key=lambda item: (item.project_id.casefold(), item.project_id, item.canonical_story_id),
        ))
        if (
            isinstance(self.aliases, (str, bytes))
            or not isinstance(self.aliases, Sequence)
            or any(not isinstance(item, StoryIdentityAlias) for item in self.aliases)
        ):
            raise TypeError("aliases must contain StoryIdentityAlias values")
        if len(self.aliases) > MAX_CANONICAL_STORY_ALIASES:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
            )
        aliases_by_id: dict[str, StoryIdentityAlias] = {}
        for alias in self.aliases:
            previous = aliases_by_id.get(alias.alias_story_id)
            if previous is not None and previous != alias:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CONFLICTING_ALIAS
                )
            source = identities_by_id.get(alias.alias_story_id)
            target = identities_by_id.get(alias.canonical_story_id)
            if source is None or target is None:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CONFLICTING_ALIAS
                )
            if source.project_id != alias.project_id or target.project_id != alias.project_id:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CROSS_PROJECT_ALIAS
                )
            aliases_by_id[alias.alias_story_id] = alias
        flattened_aliases: list[StoryIdentityAlias] = []
        for alias_id in sorted(aliases_by_id):
            target_id = _resolve_alias_target(alias_id, aliases_by_id)
            flattened_aliases.append(StoryIdentityAlias(
                project_id=aliases_by_id[alias_id].project_id,
                alias_story_id=alias_id,
                canonical_story_id=target_id,
            ))
        normalized_aliases = {
            item.alias_story_id: item for item in flattened_aliases
        }
        if (
            isinstance(self.candidate_links, (str, bytes))
            or not isinstance(self.candidate_links, Sequence)
            or any(
                not isinstance(item, CandidateCanonicalStoryLink)
                for item in self.candidate_links
            )
        ):
            raise TypeError("candidate_links must contain candidate links")
        links_by_key: dict[tuple[str, str], CandidateCanonicalStoryLink] = {}
        for link in self.candidate_links:
            target_id = _resolve_alias_target(
                link.canonical_story_id,
                normalized_aliases,
            )
            target = identities_by_id.get(target_id)
            if target is None or target.project_id != link.project_id:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CONFLICTING_CANDIDATE_LINK
                )
            normalized = CandidateCanonicalStoryLink(
                project_id=link.project_id,
                candidate_story_id=link.candidate_story_id,
                canonical_story_id=target_id,
            )
            key = (normalized.project_id, normalized.candidate_story_id)
            previous = links_by_key.get(key)
            if previous is not None and previous != normalized:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.CONFLICTING_CANDIDATE_LINK
                )
            links_by_key[key] = normalized
        if len(links_by_key) > MAX_CANONICALIZATION_DECISIONS:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
            )
        links = tuple(sorted(
            links_by_key.values(),
            key=lambda item: (
                item.project_id.casefold(),
                item.project_id,
                item.candidate_story_id,
            ),
        ))
        if (
            isinstance(self.split_relationships, (str, bytes))
            or not isinstance(self.split_relationships, Sequence)
            or any(
                not isinstance(item, StorySplitRelationship)
                for item in self.split_relationships
            )
        ):
            raise TypeError("split_relationships must contain split relationships")
        split_by_key: dict[tuple[str, str, tuple[str, ...]], StorySplitRelationship] = {}
        for relationship in self.split_relationships:
            parent = identities_by_id.get(relationship.parent_canonical_story_id)
            if parent is None or parent.project_id != relationship.project_id:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
                )
            for child_id in relationship.child_canonical_story_ids:
                child = identities_by_id.get(child_id)
                if child is None or child.project_id != relationship.project_id:
                    raise EngineeringStoryMatchingError(
                        EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
                    )
            key = (
                relationship.project_id,
                relationship.parent_canonical_story_id,
                relationship.child_candidate_story_ids,
            )
            previous = split_by_key.get(key)
            if previous is not None and previous != relationship:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.INVALID_SPLIT_RELATIONSHIP
                )
            split_by_key[key] = relationship
        splits = tuple(sorted(
            split_by_key.values(),
            key=lambda item: (
                item.project_id.casefold(),
                item.project_id,
                item.parent_canonical_story_id,
                item.child_candidate_story_ids,
            ),
        ))
        object.__setattr__(self, "canonical_identities", identities)
        object.__setattr__(self, "candidate_links", links)
        object.__setattr__(self, "aliases", tuple(flattened_aliases))
        object.__setattr__(self, "split_relationships", splits)


def resolve_canonical_story_alias(
    identity_map: EngineeringStoryIdentityMap,
    canonical_story_id: str,
) -> str:
    if not isinstance(identity_map, EngineeringStoryIdentityMap):
        raise TypeError("identity_map must be an EngineeringStoryIdentityMap")
    story_id = _canonical_story_id(canonical_story_id)
    identities = {
        item.canonical_story_id for item in identity_map.canonical_identities
    }
    if story_id not in identities:
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.CONFLICTING_IDENTITY
        )
    aliases = {item.alias_story_id: item for item in identity_map.aliases}
    return _resolve_alias_target(story_id, aliases)


@dataclass(frozen=True, slots=True)
class StoryMatchDecision(EngineeringStoryContract):
    project_id: str
    candidate_story_id: str
    outcome: StoryMatchOutcome
    canonical_story_id: str | None
    compared_canonical_story_ids: tuple[str, ...]
    signals: tuple[StoryMatchSignal, ...]
    reason_code: StoryMatchReasonCode

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        candidate_id = _candidate_story_id(self.candidate_story_id)
        outcome = StoryMatchOutcome(self.outcome)
        canonical_id = (
            None
            if self.canonical_story_id is None
            else _canonical_story_id(self.canonical_story_id)
        )
        compared = _stable_ids(
            self.compared_canonical_story_ids,
            _canonical_story_id,
            maximum=MAX_MERGE_PARTICIPANTS,
            name="compared_canonical_story_ids",
        )
        signals = _stable_enums(
            self.signals,
            StoryMatchSignal,
            maximum=len(StoryMatchSignal),
            name="signals",
        )
        reason = StoryMatchReasonCode(self.reason_code)
        if outcome in {
            StoryMatchOutcome.EXACT_MATCH,
            StoryMatchOutcome.STRONG_MATCH,
            StoryMatchOutcome.NEW_CANONICAL,
            StoryMatchOutcome.MERGE_CANDIDATE,
        }:
            if canonical_id is None:
                raise ValueError("resolved match outcome requires canonical identity")
        elif canonical_id is not None:
            raise ValueError("unresolved match outcome cannot choose canonical identity")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "candidate_story_id", candidate_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "compared_canonical_story_ids", compared)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "reason_code", reason)


@dataclass(frozen=True, slots=True)
class StoryMergeDecision(EngineeringStoryContract):
    project_id: str
    triggering_candidate_story_id: str
    participant_canonical_story_ids: tuple[str, ...]
    survivor_canonical_story_id: str | None
    aliased_canonical_story_ids: tuple[str, ...]
    outcome: StoryMergeOutcome
    reason_code: StoryMatchReasonCode

    def __post_init__(self) -> None:
        project_id = _exact_project_id(self.project_id)
        candidate_id = _candidate_story_id(self.triggering_candidate_story_id)
        participants = _stable_ids(
            self.participant_canonical_story_ids,
            _canonical_story_id,
            maximum=MAX_MERGE_PARTICIPANTS,
            name="participant_canonical_story_ids",
        )
        aliases = _stable_ids(
            self.aliased_canonical_story_ids,
            _canonical_story_id,
            maximum=MAX_MERGE_PARTICIPANTS,
            name="aliased_canonical_story_ids",
        )
        outcome = StoryMergeOutcome(self.outcome)
        survivor = (
            None
            if self.survivor_canonical_story_id is None
            else _canonical_story_id(self.survivor_canonical_story_id)
        )
        if len(participants) < 2:
            raise ValueError("merge decision requires multiple participants")
        if outcome is StoryMergeOutcome.MERGED:
            if (
                survivor not in participants
                or aliases != tuple(item for item in participants if item != survivor)
            ):
                raise ValueError("merged decision requires one deterministic survivor")
        elif survivor is not None or aliases:
            raise ValueError("unresolved merge cannot create survivor or aliases")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "triggering_candidate_story_id", candidate_id)
        object.__setattr__(self, "participant_canonical_story_ids", participants)
        object.__setattr__(self, "survivor_canonical_story_id", survivor)
        object.__setattr__(self, "aliased_canonical_story_ids", aliases)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reason_code", StoryMatchReasonCode(self.reason_code))


@dataclass(frozen=True, slots=True)
class StorySplitDecision(EngineeringStoryContract):
    project_id: str
    parent_canonical_story_id: str
    child_candidate_story_ids: tuple[str, ...]
    retained_parent_candidate_story_id: str | None
    new_child_canonical_story_ids: tuple[str, ...]
    outcome: StorySplitOutcome
    reason_code: StoryMatchReasonCode

    def __post_init__(self) -> None:
        relationship = StorySplitRelationship(
            project_id=self.project_id,
            parent_canonical_story_id=self.parent_canonical_story_id,
            child_candidate_story_ids=self.child_candidate_story_ids,
            child_canonical_story_ids=(
                ()
                if StorySplitOutcome(self.outcome) is StorySplitOutcome.AMBIGUOUS
                else (
                    self.parent_canonical_story_id,
                    *self.new_child_canonical_story_ids,
                )
            ),
            retained_parent_candidate_story_id=self.retained_parent_candidate_story_id,
            outcome=self.outcome,
        )
        object.__setattr__(self, "project_id", relationship.project_id)
        object.__setattr__(
            self,
            "parent_canonical_story_id",
            relationship.parent_canonical_story_id,
        )
        object.__setattr__(
            self,
            "child_candidate_story_ids",
            relationship.child_candidate_story_ids,
        )
        object.__setattr__(
            self,
            "retained_parent_candidate_story_id",
            relationship.retained_parent_candidate_story_id,
        )
        new_children = _stable_ids(
            self.new_child_canonical_story_ids,
            _canonical_story_id,
            maximum=MAX_SPLIT_CHILDREN,
            name="new_child_canonical_story_ids",
        )
        if relationship.parent_canonical_story_id in new_children:
            raise ValueError("parent identity is not a new split child")
        if relationship.outcome is StorySplitOutcome.AMBIGUOUS and new_children:
            raise ValueError("ambiguous split cannot assign child identities")
        object.__setattr__(self, "new_child_canonical_story_ids", new_children)
        object.__setattr__(self, "outcome", relationship.outcome)
        object.__setattr__(self, "reason_code", StoryMatchReasonCode(self.reason_code))


def _result_payload(result: "StoryCanonicalizationResult") -> dict[str, Any]:
    return {
        "identity_map": result.identity_map.to_dict(),
        "match_decisions": [item.to_dict() for item in result.match_decisions],
        "merge_decisions": [item.to_dict() for item in result.merge_decisions],
        "split_decisions": [item.to_dict() for item in result.split_decisions],
        "diagnostics": [item.value for item in result.diagnostics],
    }


@dataclass(frozen=True, slots=True)
class StoryCanonicalizationResult(EngineeringStoryContract):
    identity_map: EngineeringStoryIdentityMap
    match_decisions: tuple[StoryMatchDecision, ...]
    merge_decisions: tuple[StoryMergeDecision, ...]
    split_decisions: tuple[StorySplitDecision, ...]
    diagnostics: tuple[StoryCanonicalizationDiagnosticCode, ...]
    result_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.identity_map, EngineeringStoryIdentityMap):
            raise TypeError("identity_map must be an EngineeringStoryIdentityMap")
        typed_specs = (
            ("match_decisions", self.match_decisions, StoryMatchDecision),
            ("merge_decisions", self.merge_decisions, StoryMergeDecision),
            ("split_decisions", self.split_decisions, StorySplitDecision),
        )
        normalized: dict[str, tuple[Any, ...]] = {}
        for name, values, item_type in typed_specs:
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, Sequence)
                or any(not isinstance(item, item_type) for item in values)
            ):
                raise TypeError(f"{name} contains invalid values")
            if len(values) > MAX_CANONICALIZATION_DECISIONS:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
                )
            normalized[name] = tuple(sorted(
                set(values),
                key=lambda item: _canonical_json(item.to_dict()),
            ))
        diagnostics = _stable_enums(
            self.diagnostics,
            StoryCanonicalizationDiagnosticCode,
            maximum=MAX_CANONICALIZATION_DIAGNOSTICS,
            name="diagnostics",
        )
        object.__setattr__(self, "match_decisions", normalized["match_decisions"])
        object.__setattr__(self, "merge_decisions", normalized["merge_decisions"])
        object.__setattr__(self, "split_decisions", normalized["split_decisions"])
        object.__setattr__(self, "diagnostics", diagnostics)
        expected = _fingerprint(_result_payload(self))
        if self.result_fingerprint not in ("", expected):
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.CONFLICTING_IDENTITY
            )
        object.__setattr__(self, "result_fingerprint", expected)


@dataclass(frozen=True, slots=True)
class _PairMatch:
    signals: tuple[StoryMatchSignal, ...]
    exact_candidate: bool
    same_durable_core: bool
    strong_provenance: bool
    full_provenance_containment: bool

    @property
    def strong(self) -> bool:
        return self.exact_candidate or self.same_durable_core or self.strong_provenance


def _pair_match(
    candidate: EngineeringStoryMemoryRecord,
    existing: EngineeringStoryMemoryRecord,
) -> _PairMatch:
    if candidate.project_id != existing.project_id:
        return _PairMatch((), False, False, False, False)
    exact_candidate = candidate.candidate_story_id == existing.candidate_story_id
    same_core = (
        candidate.identity.cluster_identity_state
        is StoryClusterIdentityState.STABLE_EVENT_CORE
        and existing.identity.cluster_identity_state
        is StoryClusterIdentityState.STABLE_EVENT_CORE
        and candidate.identity.event_core_fingerprint
        == existing.identity.event_core_fingerprint
    )
    evidence_overlap = bool(
        set(candidate.provenance.evidence_fact_ids)
        & set(existing.provenance.evidence_fact_ids)
    )
    source_overlap = bool(
        set(candidate.provenance.source_lineage_fingerprints)
        & set(existing.provenance.source_lineage_fingerprints)
    )
    signals: list[StoryMatchSignal] = []
    if exact_candidate:
        signals.append(StoryMatchSignal.EXACT_CANDIDATE_IDENTITY)
    if same_core:
        signals.append(StoryMatchSignal.SAME_DURABLE_EVENT_CORE)
    if evidence_overlap:
        signals.append(StoryMatchSignal.EVIDENCE_AUTHORITY_CONTINUITY)
    if source_overlap:
        signals.append(StoryMatchSignal.SOURCE_LINEAGE_CONTINUITY)
    if candidate.engineering_story.story_type is existing.engineering_story.story_type:
        signals.append(StoryMatchSignal.SAME_STORY_TYPE)
    if set(candidate.provenance.capability_fact_ids) & set(
        existing.provenance.capability_fact_ids
    ):
        signals.append(StoryMatchSignal.CAPABILITY_CONTEXT_OVERLAP)
    evidence_contained = bool(existing.provenance.evidence_fact_ids) and set(
        existing.provenance.evidence_fact_ids
    ).issubset(candidate.provenance.evidence_fact_ids)
    source_contained = bool(existing.provenance.source_lineage_fingerprints) and set(
        existing.provenance.source_lineage_fingerprints
    ).issubset(candidate.provenance.source_lineage_fingerprints)
    return _PairMatch(
        tuple(signals),
        exact_candidate,
        same_core,
        evidence_overlap and source_overlap,
        evidence_contained and source_contained,
    )


def _identity_for_record(
    record: EngineeringStoryMemoryRecord,
) -> CanonicalEngineeringStoryIdentity:
    if (
        record.identity.cluster_identity_state
        is StoryClusterIdentityState.STABLE_EVENT_CORE
    ):
        kind = CanonicalStorySeedKind.STABLE_EVENT_CORE
        seed = record.identity.event_core_fingerprint
    else:
        kind = CanonicalStorySeedKind.PROVISIONAL_FOUNDING_IDENTITY
        seed = record.identity.identity_basis_fingerprint
    return CanonicalEngineeringStoryIdentity(
        project_id=record.project_id,
        canonical_story_id=build_canonical_engineering_story_id(
            project_id=record.project_id,
            seed_kind=kind,
            seed_fingerprint=seed,
        ),
        founding_seed_kind=kind,
        founding_seed_fingerprint=seed,
    )


def _record_sort_key(record: EngineeringStoryMemoryRecord) -> tuple[str, ...]:
    return (
        record.project_id.casefold(),
        record.project_id,
        record.candidate_story_id,
        record.record_fingerprint,
    )


def _new_record_values(
    values: Sequence[EngineeringStoryMemoryRecord],
) -> tuple[EngineeringStoryMemoryRecord, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("new_records must be a sequence")
    if len(values) > MAX_CANONICALIZATION_CANDIDATES:
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
        )
    if any(not isinstance(item, EngineeringStoryMemoryRecord) for item in values):
        raise TypeError("new_records must contain EngineeringStoryMemoryRecord values")
    by_candidate: dict[tuple[str, str], EngineeringStoryMemoryRecord] = {}
    for record in values:
        key = (record.project_id, record.candidate_story_id)
        previous = by_candidate.get(key)
        if previous is not None and previous != record:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.CONFLICTING_CANDIDATE_LINK
            )
        by_candidate[key] = record
    return tuple(sorted(by_candidate.values(), key=_record_sort_key))


def _representatives(
    memory: EngineeringStoryMemory,
    links: Mapping[tuple[str, str], str],
    aliases: Mapping[str, str],
) -> dict[str, list[EngineeringStoryMemoryRecord]]:
    result: dict[str, list[EngineeringStoryMemoryRecord]] = {}
    for record in memory.records:
        canonical_id = links.get((record.project_id, record.candidate_story_id))
        if canonical_id is None:
            continue
        while canonical_id in aliases:
            canonical_id = aliases[canonical_id]
        result.setdefault(canonical_id, []).append(record)
    for values in result.values():
        values.sort(key=_record_sort_key)
    return result


def _matches_by_identity(
    candidate: EngineeringStoryMemoryRecord,
    representatives: Mapping[str, Sequence[EngineeringStoryMemoryRecord]],
    identities: Mapping[str, CanonicalEngineeringStoryIdentity],
    comparisons: list[int],
) -> dict[str, _PairMatch]:
    matches: dict[str, _PairMatch] = {}
    for canonical_id in sorted(representatives):
        identity = identities[canonical_id]
        if identity.project_id != candidate.project_id:
            continue
        best: _PairMatch | None = None
        for representative in representatives[canonical_id]:
            comparisons[0] += 1
            if comparisons[0] > MAX_CANONICAL_COMPARISONS:
                raise EngineeringStoryMatchingError(
                    EngineeringStoryMatchingErrorCode.COMPARISON_BOUND_EXCEEDED
                )
            current = _pair_match(candidate, representative)
            if not current.strong:
                continue
            if best is None or (
                current.exact_candidate,
                current.same_durable_core,
                current.full_provenance_containment,
                len(current.signals),
            ) > (
                best.exact_candidate,
                best.same_durable_core,
                best.full_provenance_containment,
                len(best.signals),
            ):
                best = current
        if best is not None:
            matches[canonical_id] = best
    return matches


def _context_signals(
    candidate: EngineeringStoryMemoryRecord,
    representatives: Mapping[str, Sequence[EngineeringStoryMemoryRecord]],
) -> tuple[StoryMatchSignal, ...]:
    signals: set[StoryMatchSignal] = set()
    for values in representatives.values():
        for existing in values:
            if candidate.project_id != existing.project_id:
                continue
            pair = _pair_match(candidate, existing)
            signals.update(
                signal
                for signal in pair.signals
                if signal in {
                    StoryMatchSignal.SAME_STORY_TYPE,
                    StoryMatchSignal.CAPABILITY_CONTEXT_OVERLAP,
                }
            )
    return _stable_enums(
        tuple(signals),
        StoryMatchSignal,
        maximum=len(StoryMatchSignal),
        name="context_signals",
    )


def _is_partition(
    parent: EngineeringStoryMemoryRecord,
    children: Sequence[EngineeringStoryMemoryRecord],
) -> bool:
    if len(children) < 2 or len(children) > MAX_SPLIT_CHILDREN:
        return False
    if len({item.identity.event_core_fingerprint for item in children}) < 2:
        return False
    evidence_sets = [set(item.provenance.evidence_fact_ids) for item in children]
    source_sets = [
        set(item.provenance.source_lineage_fingerprints) for item in children
    ]
    if any(not values for values in evidence_sets + source_sets):
        return False
    for index, evidence in enumerate(evidence_sets):
        for other in evidence_sets[index + 1:]:
            if evidence & other:
                return False
    for index, sources in enumerate(source_sets):
        for other in source_sets[index + 1:]:
            if sources & other:
                return False
    return (
        set().union(*evidence_sets) == set(parent.provenance.evidence_fact_ids)
        and set().union(*source_sets)
        == set(parent.provenance.source_lineage_fingerprints)
    )


def canonicalize_engineering_story_memory(
    *,
    existing_memory: EngineeringStoryMemory,
    new_records: Sequence[EngineeringStoryMemoryRecord] = (),
    existing_identity_map: EngineeringStoryIdentityMap | None = None,
) -> StoryCanonicalizationResult:
    """Resolve bounded provisional candidates without rewriting their content."""

    if not isinstance(existing_memory, EngineeringStoryMemory):
        raise TypeError("existing_memory must be an EngineeringStoryMemory")
    incoming = _new_record_values(new_records)
    identity_map = existing_identity_map or EngineeringStoryIdentityMap()
    if not isinstance(identity_map, EngineeringStoryIdentityMap):
        raise TypeError("existing_identity_map must be EngineeringStoryIdentityMap")
    identities = {
        item.canonical_story_id: item
        for item in identity_map.canonical_identities
    }
    aliases = {
        item.alias_story_id: item.canonical_story_id
        for item in identity_map.aliases
    }
    links = {
        (item.project_id, item.candidate_story_id): item.canonical_story_id
        for item in identity_map.candidate_links
    }
    splits = list(identity_map.split_relationships)
    decisions: list[StoryMatchDecision] = []
    merge_decisions: list[StoryMergeDecision] = []
    split_decisions: list[StorySplitDecision] = []
    diagnostics: set[StoryCanonicalizationDiagnosticCode] = set()
    comparisons = [0]

    def active_representatives() -> dict[str, list[EngineeringStoryMemoryRecord]]:
        return _representatives(existing_memory, links, aliases)

    def add_identity(record: EngineeringStoryMemoryRecord) -> str:
        identity = _identity_for_record(record)
        previous = identities.get(identity.canonical_story_id)
        if previous is not None and previous != identity:
            raise EngineeringStoryMatchingError(
                EngineeringStoryMatchingErrorCode.CONFLICTING_IDENTITY
            )
        identities[identity.canonical_story_id] = identity
        links[(record.project_id, record.candidate_story_id)] = (
            identity.canonical_story_id
        )
        return identity.canonical_story_id

    # Seed any previously stored candidate that has not yet been canonicalized.
    for record in existing_memory.records:
        key = (record.project_id, record.candidate_story_id)
        if key in links:
            continue
        representatives = active_representatives()
        matches = _matches_by_identity(
            record,
            representatives,
            identities,
            comparisons,
        )
        matches = {
            canonical_id: pair
            for canonical_id, pair in matches.items()
            if pair.exact_candidate or pair.same_durable_core
        }
        if len(matches) == 1:
            canonical_id = next(iter(matches))
            links[key] = canonical_id
        else:
            add_identity(record)

    processed: set[tuple[str, str]] = set()
    representatives = active_representatives()
    incoming_matches: dict[tuple[str, str], dict[str, _PairMatch]] = {}
    for record in incoming:
        key = (record.project_id, record.candidate_story_id)
        if key in links:
            continue
        incoming_matches[key] = _matches_by_identity(
            record,
            representatives,
            identities,
            comparisons,
        )

    # Detect an old canonical event that has become disjoint child events.
    children_by_parent: dict[str, list[EngineeringStoryMemoryRecord]] = {}
    for record in incoming:
        key = (record.project_id, record.candidate_story_id)
        matches = incoming_matches.get(key, {})
        if len(matches) == 1:
            parent_id = next(iter(matches))
            children_by_parent.setdefault(parent_id, []).append(record)
    for parent_id in sorted(children_by_parent):
        children = tuple(sorted(children_by_parent[parent_id], key=_record_sort_key))
        parent_records = representatives.get(parent_id, ())
        parent = parent_records[0] if parent_records else None
        if parent is None or not _is_partition(parent, children):
            continue
        owners = [
            child
            for child in children
            if child.identity.event_core_fingerprint
            == parent.identity.event_core_fingerprint
        ]
        child_ids = tuple(item.candidate_story_id for item in children)
        if len(owners) != 1:
            split = StorySplitRelationship(
                project_id=parent.project_id,
                parent_canonical_story_id=parent_id,
                child_candidate_story_ids=child_ids,
                child_canonical_story_ids=(),
                retained_parent_candidate_story_id=None,
                outcome=StorySplitOutcome.AMBIGUOUS,
            )
            splits.append(split)
            split_decisions.append(StorySplitDecision(
                project_id=parent.project_id,
                parent_canonical_story_id=parent_id,
                child_candidate_story_ids=child_ids,
                retained_parent_candidate_story_id=None,
                new_child_canonical_story_ids=(),
                outcome=StorySplitOutcome.AMBIGUOUS,
                reason_code=StoryMatchReasonCode.SPLIT_AMBIGUOUS_PARENT,
            ))
            diagnostics.add(
                StoryCanonicalizationDiagnosticCode.AMBIGUOUS_SPLIT_PRESERVED
            )
            for child in children:
                processed.add((child.project_id, child.candidate_story_id))
                decisions.append(StoryMatchDecision(
                    project_id=child.project_id,
                    candidate_story_id=child.candidate_story_id,
                    outcome=StoryMatchOutcome.SPLIT_CANDIDATE,
                    canonical_story_id=None,
                    compared_canonical_story_ids=(parent_id,),
                    signals=incoming_matches[
                        (child.project_id, child.candidate_story_id)
                    ][parent_id].signals,
                    reason_code=StoryMatchReasonCode.SPLIT_AMBIGUOUS_PARENT,
                ))
            continue
        owner = owners[0]
        links[(owner.project_id, owner.candidate_story_id)] = parent_id
        new_child_ids: list[str] = []
        for child in children:
            if child is owner:
                continue
            new_child_ids.append(add_identity(child))
        all_child_ids = tuple(sorted((parent_id, *new_child_ids)))
        split = StorySplitRelationship(
            project_id=parent.project_id,
            parent_canonical_story_id=parent_id,
            child_candidate_story_ids=child_ids,
            child_canonical_story_ids=all_child_ids,
            retained_parent_candidate_story_id=owner.candidate_story_id,
            outcome=StorySplitOutcome.SPLIT,
        )
        splits.append(split)
        split_decisions.append(StorySplitDecision(
            project_id=parent.project_id,
            parent_canonical_story_id=parent_id,
            child_candidate_story_ids=child_ids,
            retained_parent_candidate_story_id=owner.candidate_story_id,
            new_child_canonical_story_ids=tuple(new_child_ids),
            outcome=StorySplitOutcome.SPLIT,
            reason_code=StoryMatchReasonCode.SPLIT_DISTINCT_EVENT_CORES,
        ))
        for child in children:
            child_canonical_id = links[(child.project_id, child.candidate_story_id)]
            processed.add((child.project_id, child.candidate_story_id))
            decisions.append(StoryMatchDecision(
                project_id=child.project_id,
                candidate_story_id=child.candidate_story_id,
                outcome=StoryMatchOutcome.SPLIT_CANDIDATE,
                canonical_story_id=None,
                compared_canonical_story_ids=(parent_id,),
                signals=incoming_matches[
                    (child.project_id, child.candidate_story_id)
                ][parent_id].signals,
                reason_code=StoryMatchReasonCode.SPLIT_DISTINCT_EVENT_CORES,
            ))
            if child is not owner:
                decisions.append(StoryMatchDecision(
                    project_id=child.project_id,
                    candidate_story_id=child.candidate_story_id,
                    outcome=StoryMatchOutcome.NEW_CANONICAL,
                    canonical_story_id=child_canonical_id,
                    compared_canonical_story_ids=(parent_id,),
                    signals=(),
                    reason_code=StoryMatchReasonCode.NEW_DISTINCT_EVENT,
                ))

    for record in incoming:
        key = (record.project_id, record.candidate_story_id)
        if key in processed:
            continue
        if key in links:
            canonical_id = links[key]
            decisions.append(StoryMatchDecision(
                project_id=record.project_id,
                candidate_story_id=record.candidate_story_id,
                outcome=StoryMatchOutcome.EXACT_MATCH,
                canonical_story_id=canonical_id,
                compared_canonical_story_ids=(canonical_id,),
                signals=(StoryMatchSignal.EXACT_CANDIDATE_IDENTITY,),
                reason_code=StoryMatchReasonCode.EXACT_CANDIDATE_MATCH,
            ))
            continue
        representatives = active_representatives()
        matches = _matches_by_identity(
            record,
            representatives,
            identities,
            comparisons,
        )
        if len(matches) == 1:
            canonical_id, pair = next(iter(matches.items()))
            links[key] = canonical_id
            decisions.append(StoryMatchDecision(
                project_id=record.project_id,
                candidate_story_id=record.candidate_story_id,
                outcome=StoryMatchOutcome.STRONG_MATCH,
                canonical_story_id=canonical_id,
                compared_canonical_story_ids=(canonical_id,),
                signals=pair.signals,
                reason_code=(
                    StoryMatchReasonCode.SAME_DURABLE_EVENT_CORE
                    if pair.same_durable_core
                    else StoryMatchReasonCode.STRONG_PROVENANCE_CONTINUITY
                ),
            ))
            continue
        if len(matches) > 1:
            participants = tuple(sorted(matches))
            safe_merge = (
                record.identity.cluster_identity_state
                is StoryClusterIdentityState.STABLE_EVENT_CORE
                and len(participants) <= MAX_MERGE_PARTICIPANTS
                and all(item.full_provenance_containment for item in matches.values())
            )
            if safe_merge:
                survivor = min(participants)
                losers = tuple(item for item in participants if item != survivor)
                for loser in losers:
                    aliases[loser] = survivor
                for link_key, target in tuple(links.items()):
                    if target in losers:
                        links[link_key] = survivor
                links[key] = survivor
                merge_decisions.append(StoryMergeDecision(
                    project_id=record.project_id,
                    triggering_candidate_story_id=record.candidate_story_id,
                    participant_canonical_story_ids=participants,
                    survivor_canonical_story_id=survivor,
                    aliased_canonical_story_ids=losers,
                    outcome=StoryMergeOutcome.MERGED,
                    reason_code=StoryMatchReasonCode.MERGE_STRONG_COMMON_CORE,
                ))
                decisions.append(StoryMatchDecision(
                    project_id=record.project_id,
                    candidate_story_id=record.candidate_story_id,
                    outcome=StoryMatchOutcome.MERGE_CANDIDATE,
                    canonical_story_id=survivor,
                    compared_canonical_story_ids=participants,
                    signals=tuple({
                        signal
                        for pair in matches.values()
                        for signal in pair.signals
                    }),
                    reason_code=StoryMatchReasonCode.MERGE_STRONG_COMMON_CORE,
                ))
            else:
                merge_decisions.append(StoryMergeDecision(
                    project_id=record.project_id,
                    triggering_candidate_story_id=record.candidate_story_id,
                    participant_canonical_story_ids=participants,
                    survivor_canonical_story_id=None,
                    aliased_canonical_story_ids=(),
                    outcome=StoryMergeOutcome.AMBIGUOUS,
                    reason_code=StoryMatchReasonCode.AMBIGUOUS_MULTI_MATCH,
                ))
                decisions.append(StoryMatchDecision(
                    project_id=record.project_id,
                    candidate_story_id=record.candidate_story_id,
                    outcome=StoryMatchOutcome.AMBIGUOUS,
                    canonical_story_id=None,
                    compared_canonical_story_ids=participants,
                    signals=tuple({
                        signal
                        for pair in matches.values()
                        for signal in pair.signals
                    }),
                    reason_code=StoryMatchReasonCode.AMBIGUOUS_MULTI_MATCH,
                ))
                diagnostics.add(
                    StoryCanonicalizationDiagnosticCode.AMBIGUOUS_MATCH_PRESERVED
                )
            continue
        context = _context_signals(record, representatives)
        canonical_id = add_identity(record)
        if context:
            diagnostics.add(
                StoryCanonicalizationDiagnosticCode.CONTEXT_ONLY_MATCH_REJECTED
            )
        if any(
            item.project_id != record.project_id for item in identities.values()
        ):
            diagnostics.add(
                StoryCanonicalizationDiagnosticCode.CROSS_PROJECT_CANDIDATE_ISOLATED
            )
        decisions.append(StoryMatchDecision(
            project_id=record.project_id,
            candidate_story_id=record.candidate_story_id,
            outcome=StoryMatchOutcome.NEW_CANONICAL,
            canonical_story_id=canonical_id,
            compared_canonical_story_ids=(),
            signals=context,
            reason_code=(
                StoryMatchReasonCode.WEAK_CONTEXT_ONLY
                if context
                else StoryMatchReasonCode.NEW_DISTINCT_EVENT
            ),
        ))

    if len(identities) > MAX_CANONICAL_STORY_IDENTITIES:
        raise EngineeringStoryMatchingError(
            EngineeringStoryMatchingErrorCode.BOUND_EXCEEDED
        )
    aliases_models = tuple(
        StoryIdentityAlias(
            project_id=identities[alias_id].project_id,
            alias_story_id=alias_id,
            canonical_story_id=target_id,
        )
        for alias_id, target_id in sorted(aliases.items())
    )
    final_map = EngineeringStoryIdentityMap(
        canonical_identities=tuple(identities.values()),
        candidate_links=tuple(
            CandidateCanonicalStoryLink(
                project_id=project_id,
                candidate_story_id=candidate_id,
                canonical_story_id=canonical_id,
            )
            for (project_id, candidate_id), canonical_id in sorted(links.items())
        ),
        aliases=aliases_models,
        split_relationships=tuple(splits),
    )
    return StoryCanonicalizationResult(
        identity_map=final_map,
        match_decisions=tuple(decisions),
        merge_decisions=tuple(merge_decisions),
        split_decisions=tuple(split_decisions),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "MAX_ALIAS_DEPTH",
    "MAX_CANONICALIZATION_CANDIDATES",
    "MAX_CANONICALIZATION_DECISIONS",
    "MAX_CANONICAL_COMPARISONS",
    "MAX_CANONICAL_STORY_ALIASES",
    "MAX_CANONICAL_STORY_IDENTITIES",
    "MAX_MERGE_PARTICIPANTS",
    "MAX_SPLIT_CHILDREN",
    "CandidateCanonicalStoryLink",
    "CanonicalEngineeringStoryIdentity",
    "CanonicalStorySeedKind",
    "EngineeringStoryIdentityMap",
    "EngineeringStoryMatchingError",
    "EngineeringStoryMatchingErrorCode",
    "StoryCanonicalizationDiagnosticCode",
    "StoryCanonicalizationResult",
    "StoryIdentityAlias",
    "StoryMatchDecision",
    "StoryMatchOutcome",
    "StoryMatchReasonCode",
    "StoryMatchSignal",
    "StoryMatchSignalCategory",
    "StoryMergeDecision",
    "StoryMergeOutcome",
    "StorySplitDecision",
    "StorySplitOutcome",
    "StorySplitRelationship",
    "build_canonical_engineering_story_id",
    "canonicalize_engineering_story_memory",
    "classify_story_match_signal",
    "resolve_canonical_story_alias",
]
