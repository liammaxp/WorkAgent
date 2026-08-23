"""Authoritative Engineering Story Memory v2 service and read boundary.

The service orchestrates the accepted Step 2--9 semantic entrypoints.  It does
not perform retrieval, model inference, JD matching, ranking, clarification, or
resume generation.  Reads never rebuild or repair the artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, TypeVar

from backend.engineering_story_clustering import cluster_story_evidence_bundle
from backend.engineering_story_evidence import (
    resolve_story_evidence_bundle_from_memory,
)
from backend.engineering_story_lifecycle import (
    EngineeringStoryLifecycleMemory,
    EngineeringStoryRevision,
    EngineeringStoryRevisionHistory,
    apply_engineering_story_revisions,
)
from backend.engineering_story_matching import (
    EngineeringStoryIdentityMap,
    StorySplitRelationship,
    canonicalize_engineering_story_memory,
    resolve_canonical_story_alias,
)
from backend.engineering_story_memory import (
    ENGINEERING_STORY_MEMORY_SCHEMA_VERSION as CANDIDATE_MEMORY_SCHEMA_VERSION,
    MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE,
    EngineeringStoryMemory,
    EngineeringStoryMemoryRecord,
    build_engineering_story_memory,
    build_engineering_story_memory_record,
)
from backend.engineering_story_models import (
    ClaimSufficiency,
    EngineeringStory,
    EngineeringStoryContract,
    EngineeringStoryLifecycle,
    EngineeringStoryStatus,
    StoryOpportunity,
    StorySufficiency,
)
from backend.engineering_story_opportunity import (
    build_story_opportunity_project_context,
    detect_story_opportunity,
)
from backend.engineering_story_reconstruction import (
    StoryReconstructionQuality,
    reconstruct_engineering_story_from_memory,
)
from backend.engineering_story_sufficiency import (
    evaluate_engineering_story_sufficiency,
)
from backend.project_evidence_memory import (
    DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
    ProjectEvidenceMemorySnapshot,
    load_project_evidence_memory,
    validate_project_evidence_memory_snapshot,
)
from backend.project_evidence_models import PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION
from backend.project_repository_identity import normalize_project_id


AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION = (
    "engineering_story_memory.v2"
)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH = (
    ROOT_DIR / "information" / "engineering_story_memory.json"
)

MAX_AUTHORITATIVE_STORY_PROJECTS = 1_000
MAX_CANONICAL_STORIES_PER_PROJECT = 512
MAX_AUTHORITATIVE_CANONICAL_STORIES = 2_048
MAX_AUTHORITATIVE_REVISIONS = 131_072
MAX_AUTHORITATIVE_ALIASES = 1_024
MAX_AUTHORITATIVE_SPLITS = 2_048
MAX_SERVICE_DIAGNOSTICS = 32
MAX_AUTHORITATIVE_STORY_MEMORY_SERIALIZED_SIZE = (
    MAX_ENGINEERING_STORY_MEMORY_SERIALIZED_SIZE
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_STORY_ID_RE = re.compile(r"^engineering_story_[0-9a-f]{24}$")
_FORBIDDEN_SERIALIZED_KEYS = {
    "raw_patch",
    "raw_text",
    "raw_source",
    "source_document",
    "document",
    "embedding",
    "embeddings",
    "file_path",
    "filesystem_path",
    "prompt",
    "jd_context",
    "company_context",
    "resume_bullet",
}


class StoryMemoryOperation(str, Enum):
    BUILD = "build"
    REFRESH = "refresh"


class StoryMemoryServiceStage(str, Enum):
    UPSTREAM_LOAD = "upstream_load"
    EVIDENCE_RESOLUTION = "evidence_resolution"
    CLUSTERING = "clustering"
    RECONSTRUCTION = "reconstruction"
    SUFFICIENCY = "sufficiency"
    OPPORTUNITY = "opportunity"
    CANDIDATE_MEMORY = "candidate_memory"
    CANONICALIZATION = "canonicalization"
    LIFECYCLE = "lifecycle"
    INTEGRITY = "integrity"
    PERSISTENCE = "persistence"


class StoryMemoryServiceErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_UPSTREAM = "invalid_upstream"
    INVALID_EXISTING_MEMORY = "invalid_existing_memory"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_ALREADY_EXISTS = "artifact_already_exists"
    STAGE_FAILED = "stage_failed"
    INTEGRITY_FAILED = "integrity_failed"
    ATOMIC_WRITE_FAILED = "atomic_write_failed"


class StoryMemoryArtifactStatus(str, Enum):
    READY = "ready"
    EMPTY = "empty"
    MISSING = "missing"
    INVALID = "invalid"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"


class StoryMemoryWriteStatus(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


class StoryMemoryReadinessState(str, Enum):
    MISSING = "missing"
    READY = "ready"
    INVALID = "invalid"
    STALE_OR_REVALIDATION_REQUIRED = "stale_or_revalidation_required"
    CONFLICTED = "conflicted"


class StoryMemoryDiagnosticCode(str, Enum):
    NONCANONICAL_UPSTREAM_PROJECT_SKIPPED = (
        "noncanonical_upstream_project_skipped"
    )
    BLOCKED_CANDIDATE_NOT_MATERIALIZED = (
        "blocked_candidate_not_materialized"
    )
    AMBIGUOUS_CANDIDATE_NOT_MATERIALIZED = (
        "ambiguous_candidate_not_materialized"
    )
    AMBIGUOUS_CANONICAL_MATCH_NOT_MATERIALIZED = (
        "ambiguous_canonical_match_not_materialized"
    )


class EngineeringStoryMemoryServiceError(ValueError):
    """Bounded service failure that never copies upstream evidence text."""

    def __init__(
        self,
        stage: StoryMemoryServiceStage | str,
        code: StoryMemoryServiceErrorCode | str,
    ) -> None:
        self.stage = StoryMemoryServiceStage(stage)
        self.code = StoryMemoryServiceErrorCode(code)
        super().__init__(f"{self.stage.value}:{self.code.value}")


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


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _count(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} is out of bounds")
    return value


def _exact_project_id(value: Any) -> str:
    normalized = normalize_project_id(value)
    if not normalized or normalized != value:
        raise ValueError("project_id must be canonical")
    return normalized


def _canonical_story_id(value: Any) -> str:
    if not isinstance(value, str) or not _CANONICAL_STORY_ID_RE.fullmatch(value):
        raise ValueError("canonical_story_id must be canonical")
    return value


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact keys must be strings")
            if key.casefold() in _FORBIDDEN_SERIALIZED_KEYS:
                raise ValueError("forbidden artifact field")
            _reject_forbidden_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_keys(child)


@dataclass(frozen=True, slots=True)
class EngineeringStorySourceArtifact(EngineeringStoryContract):
    schema_version: str
    content_hash: str
    project_count: int
    evidence_fact_count: int
    capability_fact_count: int
    claim_boundary_count: int

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_EVIDENCE_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported source schema_version")
        object.__setattr__(
            self, "content_hash", _digest(self.content_hash, "content_hash")
        )
        object.__setattr__(
            self,
            "project_count",
            _count(
                self.project_count,
                "project_count",
                maximum=MAX_AUTHORITATIVE_STORY_PROJECTS,
            ),
        )
        for name, maximum in (
            ("evidence_fact_count", 100_000),
            ("capability_fact_count", 10_000),
            ("claim_boundary_count", 100_000),
        ):
            object.__setattr__(
                self, name, _count(getattr(self, name), name, maximum=maximum)
            )


def _source_artifact(
    snapshot: ProjectEvidenceMemorySnapshot,
) -> EngineeringStorySourceArtifact:
    validation = validate_project_evidence_memory_snapshot(snapshot)
    if not validation.valid:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.UPSTREAM_LOAD,
            StoryMemoryServiceErrorCode.INVALID_UPSTREAM,
        )
    return EngineeringStorySourceArtifact(
        schema_version=validation.schema_version,
        content_hash=validation.content_hash,
        project_count=validation.project_count,
        evidence_fact_count=validation.evidence_fact_count,
        capability_fact_count=validation.capability_fact_count,
        claim_boundary_count=validation.claim_boundary_count,
    )


def _authoritative_memory_payload(
    memory: "AuthoritativeEngineeringStoryMemory",
    *,
    include_fingerprint: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": memory.schema_version,
        "source_artifact": memory.source_artifact.to_dict(),
        "identity_map": memory.identity_map.to_dict(),
        "histories": [item.to_dict() for item in memory.histories],
    }
    if include_fingerprint:
        payload["logical_fingerprint"] = memory.logical_fingerprint
    return payload


@dataclass(frozen=True, slots=True)
class AuthoritativeEngineeringStoryMemory(EngineeringStoryContract):
    schema_version: str
    source_artifact: EngineeringStorySourceArtifact
    identity_map: EngineeringStoryIdentityMap
    histories: tuple[EngineeringStoryRevisionHistory, ...] = ()
    logical_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported authoritative Story Memory schema")
        if not isinstance(self.source_artifact, EngineeringStorySourceArtifact):
            raise TypeError("source_artifact must be EngineeringStorySourceArtifact")
        if not isinstance(self.identity_map, EngineeringStoryIdentityMap):
            raise TypeError("identity_map must be EngineeringStoryIdentityMap")
        if isinstance(self.histories, (str, bytes)) or not isinstance(
            self.histories, Sequence
        ):
            raise TypeError("histories must be a sequence")
        lifecycle = EngineeringStoryLifecycleMemory(
            identity_map=self.identity_map,
            histories=tuple(self.histories),
        )
        identities = {
            item.canonical_story_id
            for item in lifecycle.identity_map.canonical_identities
        }
        history_ids = {item.canonical_story_id for item in lifecycle.histories}
        if identities != history_ids:
            raise ValueError("canonical identity/history set mismatch")
        if len(identities) > MAX_AUTHORITATIVE_CANONICAL_STORIES:
            raise ValueError("maximum canonical story count exceeded")
        if len(lifecycle.identity_map.aliases) > MAX_AUTHORITATIVE_ALIASES:
            raise ValueError("maximum alias count exceeded")
        if len(lifecycle.identity_map.split_relationships) > MAX_AUTHORITATIVE_SPLITS:
            raise ValueError("maximum split count exceeded")
        revision_count = sum(len(item.revisions) for item in lifecycle.histories)
        if revision_count > MAX_AUTHORITATIVE_REVISIONS:
            raise ValueError("maximum revision count exceeded")
        project_counts: dict[str, int] = {}
        revision_candidates: set[tuple[str, str]] = set()
        for history in lifecycle.histories:
            project_counts[history.project_id] = (
                project_counts.get(history.project_id, 0) + 1
            )
            for revision in history.revisions:
                revision_candidates.add(
                    (revision.project_id, revision.candidate_story_id)
                )
        if len(project_counts) > MAX_AUTHORITATIVE_STORY_PROJECTS:
            raise ValueError("maximum Story Memory project count exceeded")
        if any(
            count > MAX_CANONICAL_STORIES_PER_PROJECT
            for count in project_counts.values()
        ):
            raise ValueError("maximum per-project story count exceeded")
        if any(
            (link.project_id, link.candidate_story_id) not in revision_candidates
            for link in lifecycle.identity_map.candidate_links
        ):
            raise ValueError("orphan candidate identity link")
        object.__setattr__(self, "identity_map", lifecycle.identity_map)
        object.__setattr__(self, "histories", lifecycle.histories)
        expected = _fingerprint(
            _authoritative_memory_payload(self, include_fingerprint=False)
        )
        if self.logical_fingerprint not in ("", expected):
            raise ValueError("authoritative Story Memory fingerprint mismatch")
        object.__setattr__(self, "logical_fingerprint", expected)
        payload = _authoritative_memory_payload(self, include_fingerprint=True)
        _reject_forbidden_keys(payload)
        if (
            len(_canonical_json_bytes(payload, pretty=True)) + 1
            > MAX_AUTHORITATIVE_STORY_MEMORY_SERIALIZED_SIZE
        ):
            raise ValueError("maximum Story Memory artifact size exceeded")

    @property
    def lifecycle_memory(self) -> EngineeringStoryLifecycleMemory:
        return EngineeringStoryLifecycleMemory(
            identity_map=self.identity_map,
            histories=self.histories,
        )

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
class EngineeringStoryMemoryValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    schema_version: str = ""
    logical_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class AuthoritativeStoryMemoryLoadResult:
    status: StoryMemoryArtifactStatus
    memory: AuthoritativeEngineeringStoryMemory | None
    error_code: StoryMemoryServiceErrorCode | None = None

    def __post_init__(self) -> None:
        status = StoryMemoryArtifactStatus(self.status)
        error = (
            None
            if self.error_code is None
            else StoryMemoryServiceErrorCode(self.error_code)
        )
        successful = status in {
            StoryMemoryArtifactStatus.READY,
            StoryMemoryArtifactStatus.EMPTY,
        }
        if successful != isinstance(
            self.memory, AuthoritativeEngineeringStoryMemory
        ):
            raise ValueError("load status and memory disagree")
        if successful == (error is not None):
            raise ValueError("load status and error disagree")
        if successful and (
            status is StoryMemoryArtifactStatus.EMPTY
        ) != (not self.memory.histories):
            raise ValueError("empty load status mismatch")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error_code", error)


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryWriteResult:
    status: StoryMemoryWriteStatus
    logical_fingerprint: str
    bytes_written: int
    previous_artifact_preserved: bool
    round_trip_validated: bool
    error_code: StoryMemoryServiceErrorCode | None = None

    def __post_init__(self) -> None:
        status = StoryMemoryWriteStatus(self.status)
        object.__setattr__(
            self,
            "logical_fingerprint",
            _digest(self.logical_fingerprint, "logical_fingerprint"),
        )
        _count(
            self.bytes_written,
            "bytes_written",
            maximum=MAX_AUTHORITATIVE_STORY_MEMORY_SERIALIZED_SIZE,
        )
        if not isinstance(self.previous_artifact_preserved, bool):
            raise TypeError("previous_artifact_preserved must be boolean")
        if not isinstance(self.round_trip_validated, bool):
            raise TypeError("round_trip_validated must be boolean")
        error = (
            None
            if self.error_code is None
            else StoryMemoryServiceErrorCode(self.error_code)
        )
        if (status is StoryMemoryWriteStatus.FAILED) != (error is not None):
            raise ValueError("write status and error disagree")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error_code", error)


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryReadiness:
    state: StoryMemoryReadinessState
    schema_version: str = ""
    logical_fingerprint: str = ""
    project_count: int = 0
    canonical_story_count: int = 0
    active_story_count: int = 0
    requires_revalidation_count: int = 0
    stale_story_count: int = 0
    conflicted_story_count: int = 0
    superseded_story_count: int = 0
    revision_count: int = 0
    alias_count: int = 0
    split_relationship_count: int = 0
    error_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", StoryMemoryReadinessState(self.state))
        if len(self.error_codes) > MAX_SERVICE_DIAGNOSTICS:
            raise ValueError("too many readiness errors")
        object.__setattr__(self, "error_codes", tuple(sorted(set(self.error_codes))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "schema_version": self.schema_version,
            "logical_fingerprint": self.logical_fingerprint,
            "project_count": self.project_count,
            "canonical_story_count": self.canonical_story_count,
            "active_story_count": self.active_story_count,
            "requires_revalidation_count": self.requires_revalidation_count,
            "stale_story_count": self.stale_story_count,
            "conflicted_story_count": self.conflicted_story_count,
            "superseded_story_count": self.superseded_story_count,
            "revision_count": self.revision_count,
            "alias_count": self.alias_count,
            "split_relationship_count": self.split_relationship_count,
            "error_codes": list(self.error_codes),
        }


@dataclass(frozen=True, slots=True)
class EngineeringStoryMemoryOperationResult:
    operation: StoryMemoryOperation
    memory: AuthoritativeEngineeringStoryMemory
    upstream_project_count: int
    skipped_noncanonical_project_count: int
    provisional_cluster_count: int
    candidate_count: int
    unmaterialized_candidate_count: int
    canonical_story_count: int
    active_count: int
    requires_revalidation_count: int
    stale_count: int
    conflicted_count: int
    superseded_count: int
    revision_count: int
    alias_count: int
    split_count: int
    unchanged_canonical_count: int = 0
    updated_canonical_count: int = 0
    new_canonical_count: int = 0
    new_revision_count: int = 0
    aliases_added: int = 0
    splits_added: int = 0
    diagnostics: tuple[StoryMemoryDiagnosticCode, ...] = ()
    write_status: StoryMemoryWriteStatus | None = None
    bytes_written: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.memory, AuthoritativeEngineeringStoryMemory):
            raise TypeError("memory must be authoritative Story Memory")
        object.__setattr__(self, "operation", StoryMemoryOperation(self.operation))
        diagnostics = tuple(sorted(
            {StoryMemoryDiagnosticCode(item) for item in self.diagnostics},
            key=lambda item: item.value,
        ))
        if len(diagnostics) > MAX_SERVICE_DIAGNOSTICS:
            raise ValueError("too many operation diagnostics")
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.write_status is not None:
            object.__setattr__(
                self, "write_status", StoryMemoryWriteStatus(self.write_status)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "schema_version": self.memory.schema_version,
            "source_schema_version": self.memory.source_artifact.schema_version,
            "source_content_hash": self.memory.source_artifact.content_hash,
            "logical_fingerprint": self.memory.logical_fingerprint,
            "upstream_project_count": self.upstream_project_count,
            "skipped_noncanonical_project_count": self.skipped_noncanonical_project_count,
            "provisional_cluster_count": self.provisional_cluster_count,
            "candidate_count": self.candidate_count,
            "unmaterialized_candidate_count": self.unmaterialized_candidate_count,
            "canonical_story_count": self.canonical_story_count,
            "active_count": self.active_count,
            "requires_revalidation_count": self.requires_revalidation_count,
            "stale_count": self.stale_count,
            "conflicted_count": self.conflicted_count,
            "superseded_count": self.superseded_count,
            "revision_count": self.revision_count,
            "alias_count": self.alias_count,
            "split_count": self.split_count,
            "unchanged_canonical_count": self.unchanged_canonical_count,
            "updated_canonical_count": self.updated_canonical_count,
            "new_canonical_count": self.new_canonical_count,
            "new_revision_count": self.new_revision_count,
            "aliases_added": self.aliases_added,
            "splits_added": self.splits_added,
            "diagnostics": [item.value for item in self.diagnostics],
            "write_status": (
                None if self.write_status is None else self.write_status.value
            ),
            "bytes_written": self.bytes_written,
        }


@dataclass(frozen=True, slots=True)
class EngineeringStoryView(EngineeringStoryContract):
    canonical_story_id: str
    project_id: str
    current_story: EngineeringStory
    claim_sufficiency: ClaimSufficiency
    story_sufficiency: StorySufficiency
    opportunity: StoryOpportunity
    lifecycle: EngineeringStoryLifecycle
    current_revision_id: str
    evidence_fact_ids: tuple[str, ...]
    capability_fact_ids: tuple[str, ...]
    claim_boundary_ids: tuple[str, ...]
    provenance_fingerprint: str
    source_lineage_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical_id = _canonical_story_id(self.canonical_story_id)
        project_id = _exact_project_id(self.project_id)
        if not isinstance(self.current_story, EngineeringStory):
            raise TypeError("current_story must be EngineeringStory")
        if (
            self.current_story.story_id != canonical_id
            or self.current_story.project_id != project_id
            or self.current_story.lifecycle != self.lifecycle
            or self.current_story.claim_sufficiency != self.claim_sufficiency
            or self.current_story.story_sufficiency != self.story_sufficiency
            or self.current_story.opportunity != self.opportunity
        ):
            raise ValueError("Story View semantic mismatch")
        object.__setattr__(self, "canonical_story_id", canonical_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(
            self,
            "provenance_fingerprint",
            _digest(self.provenance_fingerprint, "provenance_fingerprint"),
        )


_T = TypeVar("_T")


def _run_stage(stage: StoryMemoryServiceStage, call: Callable[[], _T]) -> _T:
    try:
        return call()
    except EngineeringStoryMemoryServiceError:
        raise
    except Exception as exc:
        raise EngineeringStoryMemoryServiceError(
            stage, StoryMemoryServiceErrorCode.STAGE_FAILED
        ) from exc


@dataclass(frozen=True, slots=True)
class _CandidateBuild:
    records: tuple[EngineeringStoryMemoryRecord, ...]
    provisional_cluster_count: int
    unmaterialized_candidate_count: int
    skipped_noncanonical_project_count: int
    diagnostics: tuple[StoryMemoryDiagnosticCode, ...]


def _build_candidate_records(
    snapshot: ProjectEvidenceMemorySnapshot,
) -> _CandidateBuild:
    records: list[EngineeringStoryMemoryRecord] = []
    clusters_seen = 0
    rejected = 0
    skipped_projects = 0
    diagnostics: set[StoryMemoryDiagnosticCode] = set()
    for project_memory in snapshot.projects:
        if normalize_project_id(project_memory.project_id) != project_memory.project_id:
            skipped_projects += 1
            diagnostics.add(
                StoryMemoryDiagnosticCode.NONCANONICAL_UPSTREAM_PROJECT_SKIPPED
            )
            continue
        if not project_memory.evidence_facts:
            continue
        bundle = _run_stage(
            StoryMemoryServiceStage.EVIDENCE_RESOLUTION,
            lambda project_memory=project_memory: resolve_story_evidence_bundle_from_memory(
                project_memory=project_memory,
                evidence_fact_ids=tuple(
                    item.evidence_fact_id
                    for item in project_memory.evidence_facts
                ),
                capability_ids=tuple(
                    item.capability_id
                    for item in project_memory.capability_facts
                ),
            ),
        )
        clustering = _run_stage(
            StoryMemoryServiceStage.CLUSTERING,
            lambda bundle=bundle: cluster_story_evidence_bundle(bundle),
        )
        clusters = clustering.clusters
        clusters_seen += len(clusters)
        project_context = _run_stage(
            StoryMemoryServiceStage.OPPORTUNITY,
            lambda: build_story_opportunity_project_context(
                project_id=project_memory.project_id,
                clusters=clusters,
            ),
        )
        for cluster in clusters:
            reconstruction = _run_stage(
                StoryMemoryServiceStage.RECONSTRUCTION,
                lambda cluster=cluster: reconstruct_engineering_story_from_memory(
                    cluster=cluster,
                    project_memory=project_memory,
                ),
            )
            sufficiency = _run_stage(
                StoryMemoryServiceStage.SUFFICIENCY,
                lambda reconstruction=reconstruction: evaluate_engineering_story_sufficiency(
                    reconstruction_result=reconstruction,
                    project_memory=project_memory,
                ),
            )
            opportunity = _run_stage(
                StoryMemoryServiceStage.OPPORTUNITY,
                lambda reconstruction=reconstruction, sufficiency=sufficiency, cluster=cluster: detect_story_opportunity(
                    reconstruction_result=reconstruction,
                    sufficiency_result=sufficiency,
                    story_cluster=cluster,
                    project_context=project_context,
                ),
            )
            if reconstruction.engineering_story is None:
                rejected += 1
                diagnostics.add(
                    StoryMemoryDiagnosticCode.BLOCKED_CANDIDATE_NOT_MATERIALIZED
                )
                continue
            if reconstruction.reconstruction_quality is StoryReconstructionQuality.AMBIGUOUS:
                rejected += 1
                diagnostics.add(
                    StoryMemoryDiagnosticCode.AMBIGUOUS_CANDIDATE_NOT_MATERIALIZED
                )
                continue
            record = _run_stage(
                StoryMemoryServiceStage.CANDIDATE_MEMORY,
                lambda cluster=cluster, reconstruction=reconstruction, sufficiency=sufficiency, opportunity=opportunity: build_engineering_story_memory_record(
                    story_cluster=cluster,
                    reconstruction_result=reconstruction,
                    sufficiency_result=sufficiency,
                    opportunity_result=opportunity,
                ),
            )
            records.append(record)
    candidate_memory = _run_stage(
        StoryMemoryServiceStage.CANDIDATE_MEMORY,
        lambda: build_engineering_story_memory(tuple(records)),
    )
    return _CandidateBuild(
        records=candidate_memory.records,
        provisional_cluster_count=clusters_seen,
        unmaterialized_candidate_count=rejected,
        skipped_noncanonical_project_count=skipped_projects,
        diagnostics=tuple(sorted(diagnostics, key=lambda item: item.value)),
    )


def _authoritative_from_lifecycle(
    *,
    snapshot: ProjectEvidenceMemorySnapshot,
    lifecycle_memory: EngineeringStoryLifecycleMemory,
) -> AuthoritativeEngineeringStoryMemory:
    return _run_stage(
        StoryMemoryServiceStage.INTEGRITY,
        lambda: AuthoritativeEngineeringStoryMemory(
            schema_version=AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION,
            source_artifact=_source_artifact(snapshot),
            identity_map=lifecycle_memory.identity_map,
            histories=lifecycle_memory.histories,
        ),
    )


def _lifecycle_counts(
    memory: AuthoritativeEngineeringStoryMemory,
) -> dict[EngineeringStoryStatus, int]:
    counts = {status: 0 for status in EngineeringStoryStatus}
    for history in memory.histories:
        counts[history.current_revision.lifecycle.status] += 1
    return counts


def _operation_result(
    *,
    operation: StoryMemoryOperation,
    memory: AuthoritativeEngineeringStoryMemory,
    candidate_build: _CandidateBuild,
    previous: AuthoritativeEngineeringStoryMemory | None = None,
) -> EngineeringStoryMemoryOperationResult:
    counts = _lifecycle_counts(memory)
    current_by_id = {
        item.canonical_story_id: item.current_revision_id
        for item in memory.histories
    }
    previous_by_id = (
        {}
        if previous is None
        else {
            item.canonical_story_id: item.current_revision_id
            for item in previous.histories
        }
    )
    common = set(current_by_id) & set(previous_by_id)
    unchanged = sum(
        current_by_id[item] == previous_by_id[item] for item in common
    )
    updated = len(common) - unchanged
    unresolved = max(
        0,
        len(candidate_build.records)
        - len({
            link.candidate_story_id
            for link in memory.identity_map.candidate_links
            if any(
                record.project_id == link.project_id
                and record.candidate_story_id == link.candidate_story_id
                for record in candidate_build.records
            )
        }),
    )
    diagnostics = set(candidate_build.diagnostics)
    if unresolved:
        diagnostics.add(
            StoryMemoryDiagnosticCode.AMBIGUOUS_CANONICAL_MATCH_NOT_MATERIALIZED
        )
    return EngineeringStoryMemoryOperationResult(
        operation=operation,
        memory=memory,
        upstream_project_count=memory.source_artifact.project_count,
        skipped_noncanonical_project_count=(
            candidate_build.skipped_noncanonical_project_count
        ),
        provisional_cluster_count=candidate_build.provisional_cluster_count,
        candidate_count=len(candidate_build.records),
        unmaterialized_candidate_count=(
            candidate_build.unmaterialized_candidate_count + unresolved
        ),
        canonical_story_count=len(memory.histories),
        active_count=counts[EngineeringStoryStatus.ACTIVE],
        requires_revalidation_count=sum(
            item.current_revision.lifecycle.requires_revalidation
            for item in memory.histories
        ),
        stale_count=counts[EngineeringStoryStatus.STALE],
        conflicted_count=counts[EngineeringStoryStatus.CONFLICTED],
        superseded_count=counts[EngineeringStoryStatus.SUPERSEDED],
        revision_count=sum(len(item.revisions) for item in memory.histories),
        alias_count=len(memory.identity_map.aliases),
        split_count=len(memory.identity_map.split_relationships),
        unchanged_canonical_count=unchanged,
        updated_canonical_count=updated,
        new_canonical_count=len(set(current_by_id) - set(previous_by_id)),
        new_revision_count=(
            sum(len(item.revisions) for item in memory.histories)
            - (
                0
                if previous is None
                else sum(len(item.revisions) for item in previous.histories)
            )
        ),
        aliases_added=(
            len(memory.identity_map.aliases)
            - (0 if previous is None else len(previous.identity_map.aliases))
        ),
        splits_added=(
            len(memory.identity_map.split_relationships)
            - (
                0
                if previous is None
                else len(previous.identity_map.split_relationships)
            )
        ),
        diagnostics=tuple(diagnostics),
    )


def build_authoritative_engineering_story_memory(
    snapshot: ProjectEvidenceMemorySnapshot,
) -> EngineeringStoryMemoryOperationResult:
    """Build the first in-memory v2 artifact from one validated snapshot."""

    source = _source_artifact(snapshot)
    candidate_build = _build_candidate_records(snapshot)
    canonicalization = _run_stage(
        StoryMemoryServiceStage.CANONICALIZATION,
        lambda: canonicalize_engineering_story_memory(
            existing_memory=build_engineering_story_memory(),
            new_records=candidate_build.records,
        ),
    )
    lifecycle = _run_stage(
        StoryMemoryServiceStage.LIFECYCLE,
        lambda: apply_engineering_story_revisions(
            canonicalization_result=canonicalization,
            new_records=candidate_build.records,
        ),
    )
    memory = _authoritative_from_lifecycle(
        snapshot=snapshot,
        lifecycle_memory=lifecycle.memory,
    )
    if memory.source_artifact != source:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.INTEGRITY,
            StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
        )
    return _operation_result(
        operation=StoryMemoryOperation.BUILD,
        memory=memory,
        candidate_build=candidate_build,
    )


def _current_candidate_memory(
    memory: AuthoritativeEngineeringStoryMemory,
) -> EngineeringStoryMemory:
    return build_engineering_story_memory(tuple(
        history.current_revision.record for history in memory.histories
    ))


def refresh_authoritative_engineering_story_memory(
    *,
    existing_memory: AuthoritativeEngineeringStoryMemory,
    snapshot: ProjectEvidenceMemorySnapshot,
    revalidate_canonical_story_ids: Sequence[str] = (),
) -> EngineeringStoryMemoryOperationResult:
    """Refresh v2 without rebuilding canonical identity or revision history."""

    if not isinstance(existing_memory, AuthoritativeEngineeringStoryMemory):
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.INTEGRITY,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    _source_artifact(snapshot)
    candidate_build = _build_candidate_records(snapshot)
    canonicalization = _run_stage(
        StoryMemoryServiceStage.CANONICALIZATION,
        lambda: canonicalize_engineering_story_memory(
            existing_memory=_current_candidate_memory(existing_memory),
            new_records=candidate_build.records,
            existing_identity_map=existing_memory.identity_map,
        ),
    )
    links = {
        (item.project_id, item.candidate_story_id): item.canonical_story_id
        for item in canonicalization.identity_map.candidate_links
    }
    present = {
        links[(item.project_id, item.candidate_story_id)]
        for item in candidate_build.records
        if (item.project_id, item.candidate_story_id) in links
    }
    missing = tuple(sorted(
        history.canonical_story_id
        for history in existing_memory.histories
        if history.canonical_story_id not in present
        and history.current_revision.lifecycle.status
        is not EngineeringStoryStatus.SUPERSEDED
    ))
    lifecycle = _run_stage(
        StoryMemoryServiceStage.LIFECYCLE,
        lambda: apply_engineering_story_revisions(
            canonicalization_result=canonicalization,
            new_records=candidate_build.records,
            existing_lifecycle_memory=existing_memory.lifecycle_memory,
            revalidate_canonical_story_ids=revalidate_canonical_story_ids,
            missing_canonical_story_ids=missing,
        ),
    )
    memory = _authoritative_from_lifecycle(
        snapshot=snapshot,
        lifecycle_memory=lifecycle.memory,
    )
    return _operation_result(
        operation=StoryMemoryOperation.REFRESH,
        memory=memory,
        candidate_build=candidate_build,
        previous=existing_memory,
    )


def validate_authoritative_engineering_story_memory(
    memory: AuthoritativeEngineeringStoryMemory,
) -> EngineeringStoryMemoryValidationResult:
    try:
        if not isinstance(memory, AuthoritativeEngineeringStoryMemory):
            raise TypeError("invalid memory type")
        payload = memory.to_dict()
        _reject_forbidden_keys(payload)
        restored = AuthoritativeEngineeringStoryMemory.from_dict(payload)
        if restored != memory:
            raise ValueError("round-trip mismatch")
    except Exception:
        return EngineeringStoryMemoryValidationResult(
            valid=False,
            errors=("invalid_authoritative_story_memory",),
        )
    return EngineeringStoryMemoryValidationResult(
        valid=True,
        schema_version=memory.schema_version,
        logical_fingerprint=memory.logical_fingerprint,
    )


def serialize_authoritative_engineering_story_memory(
    memory: AuthoritativeEngineeringStoryMemory,
) -> bytes:
    validation = validate_authoritative_engineering_story_memory(memory)
    if not validation.valid:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.INTEGRITY,
            StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
        )
    serialized = _canonical_json_bytes(memory.to_dict(), pretty=True) + b"\n"
    if len(serialized) > MAX_AUTHORITATIVE_STORY_MEMORY_SERIALIZED_SIZE:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.INTEGRITY,
            StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
        )
    return serialized


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate_json_key")
        result[key] = value
    return result


def load_authoritative_engineering_story_memory(
    path: str | Path | None = None,
) -> AuthoritativeStoryMemoryLoadResult:
    """Strictly load v2; never rebuild, repair, or fall back."""

    destination = (
        DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH
        if path is None
        else Path(path)
    )
    if not destination.exists():
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.MISSING,
            None,
            StoryMemoryServiceErrorCode.ARTIFACT_MISSING,
        )
    if destination.is_dir():
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.INVALID,
            None,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    try:
        if destination.stat().st_size > MAX_AUTHORITATIVE_STORY_MEMORY_SERIALIZED_SIZE:
            raise ValueError("artifact too large")
        payload = json.loads(
            destination.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except Exception:
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.INVALID,
            None,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    if not isinstance(payload, Mapping):
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.INVALID,
            None,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    if payload.get("schema_version") != AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION:
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.UNSUPPORTED_VERSION,
            None,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    try:
        memory = AuthoritativeEngineeringStoryMemory.from_dict(payload)
    except ValueError as exc:
        status = (
            StoryMemoryArtifactStatus.INTEGRITY_MISMATCH
            if "fingerprint" in str(exc).casefold()
            else StoryMemoryArtifactStatus.INVALID
        )
        return AuthoritativeStoryMemoryLoadResult(
            status,
            None,
            StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
        )
    except (TypeError, KeyError):
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.INVALID,
            None,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    validation = validate_authoritative_engineering_story_memory(memory)
    if not validation.valid:
        return AuthoritativeStoryMemoryLoadResult(
            StoryMemoryArtifactStatus.INVALID,
            None,
            StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
        )
    return AuthoritativeStoryMemoryLoadResult(
        StoryMemoryArtifactStatus.EMPTY if not memory.histories else StoryMemoryArtifactStatus.READY,
        memory,
    )


def _write_staged_bytes(destination: Path, serialized: bytes) -> Path:
    descriptor, staged_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".stage",
    )
    staged = Path(staged_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        return staged
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def _replace_staged_authoritative_story_memory(
    staged: Path, destination: Path
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


def _restore_previous_artifact(
    destination: Path, previous_bytes: bytes | None
) -> bool:
    try:
        if previous_bytes is None:
            destination.unlink(missing_ok=True)
            return not destination.exists()
        staged = _write_staged_bytes(destination, previous_bytes)
        try:
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)
        return destination.read_bytes() == previous_bytes
    except Exception:
        return False


def write_authoritative_engineering_story_memory(
    memory: AuthoritativeEngineeringStoryMemory,
    *,
    path: str | Path | None = None,
    operation: StoryMemoryOperation,
) -> EngineeringStoryMemoryWriteResult:
    """Atomically install build/refresh output after strict temp reload."""

    operation = StoryMemoryOperation(operation)
    destination = (
        DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH
        if path is None
        else Path(path)
    )
    serialized = serialize_authoritative_engineering_story_memory(memory)
    existed = destination.exists()
    if destination.is_dir():
        return EngineeringStoryMemoryWriteResult(
            StoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            True,
            False,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    if operation is StoryMemoryOperation.BUILD and existed:
        return EngineeringStoryMemoryWriteResult(
            StoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            True,
            False,
            StoryMemoryServiceErrorCode.ARTIFACT_ALREADY_EXISTS,
        )
    previous_bytes: bytes | None = None
    if operation is StoryMemoryOperation.REFRESH:
        if not existed:
            return EngineeringStoryMemoryWriteResult(
                StoryMemoryWriteStatus.FAILED,
                memory.logical_fingerprint,
                0,
                False,
                False,
                StoryMemoryServiceErrorCode.ARTIFACT_MISSING,
            )
        existing = load_authoritative_engineering_story_memory(destination)
        if existing.memory is None:
            return EngineeringStoryMemoryWriteResult(
                StoryMemoryWriteStatus.FAILED,
                memory.logical_fingerprint,
                0,
                True,
                False,
                StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
            )
        if existing.memory == memory:
            return EngineeringStoryMemoryWriteResult(
                StoryMemoryWriteStatus.UNCHANGED,
                memory.logical_fingerprint,
                0,
                True,
                True,
            )
        try:
            previous_bytes = destination.read_bytes()
        except OSError:
            return EngineeringStoryMemoryWriteResult(
                StoryMemoryWriteStatus.FAILED,
                memory.logical_fingerprint,
                0,
                True,
                False,
                StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
            )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = _write_staged_bytes(destination, serialized)
    except Exception:
        preserved = _restore_previous_artifact(destination, previous_bytes)
        return EngineeringStoryMemoryWriteResult(
            StoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            preserved,
            False,
            StoryMemoryServiceErrorCode.ATOMIC_WRITE_FAILED,
        )
    try:
        staged_load = load_authoritative_engineering_story_memory(staged)
        if staged_load.memory != memory:
            raise ValueError("staged round-trip failed")
        _replace_staged_authoritative_story_memory(staged, destination)
        staged = None
        _sync_parent_directory(destination.parent)
        final_load = load_authoritative_engineering_story_memory(destination)
        if final_load.memory != memory:
            preserved = _restore_previous_artifact(destination, previous_bytes)
            return EngineeringStoryMemoryWriteResult(
                StoryMemoryWriteStatus.FAILED,
                memory.logical_fingerprint,
                0,
                preserved,
                False,
                StoryMemoryServiceErrorCode.ATOMIC_WRITE_FAILED,
            )
        return EngineeringStoryMemoryWriteResult(
            StoryMemoryWriteStatus.UPDATED if existed else StoryMemoryWriteStatus.CREATED,
            memory.logical_fingerprint,
            len(serialized),
            False,
            True,
        )
    except Exception:
        preserved = _restore_previous_artifact(destination, previous_bytes)
        return EngineeringStoryMemoryWriteResult(
            StoryMemoryWriteStatus.FAILED,
            memory.logical_fingerprint,
            0,
            preserved,
            False,
            StoryMemoryServiceErrorCode.ATOMIC_WRITE_FAILED,
        )
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _load_upstream_snapshot(
    path: str | Path | None,
) -> ProjectEvidenceMemorySnapshot:
    loaded = load_project_evidence_memory(path)
    if loaded.status != "ready" or loaded.snapshot is None:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.UPSTREAM_LOAD,
            StoryMemoryServiceErrorCode.INVALID_UPSTREAM,
        )
    return loaded.snapshot


def build_and_materialize_authoritative_engineering_story_memory(
    *,
    upstream_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> EngineeringStoryMemoryOperationResult:
    snapshot = _load_upstream_snapshot(upstream_path)
    result = build_authoritative_engineering_story_memory(snapshot)
    write = write_authoritative_engineering_story_memory(
        result.memory,
        path=output_path,
        operation=StoryMemoryOperation.BUILD,
    )
    if write.status is StoryMemoryWriteStatus.FAILED:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.PERSISTENCE,
            write.error_code or StoryMemoryServiceErrorCode.ATOMIC_WRITE_FAILED,
        )
    return replace(
        result,
        write_status=write.status,
        bytes_written=write.bytes_written,
    )


def refresh_and_materialize_authoritative_engineering_story_memory(
    *,
    upstream_path: str | Path | None = None,
    output_path: str | Path | None = None,
    revalidate_canonical_story_ids: Sequence[str] = (),
) -> EngineeringStoryMemoryOperationResult:
    destination = (
        DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH
        if output_path is None
        else Path(output_path)
    )
    existing = load_authoritative_engineering_story_memory(destination)
    if existing.memory is None:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.PERSISTENCE,
            StoryMemoryServiceErrorCode.INVALID_EXISTING_MEMORY,
        )
    snapshot = _load_upstream_snapshot(upstream_path)
    result = refresh_authoritative_engineering_story_memory(
        existing_memory=existing.memory,
        snapshot=snapshot,
        revalidate_canonical_story_ids=revalidate_canonical_story_ids,
    )
    write = write_authoritative_engineering_story_memory(
        result.memory,
        path=destination,
        operation=StoryMemoryOperation.REFRESH,
    )
    if write.status is StoryMemoryWriteStatus.FAILED:
        raise EngineeringStoryMemoryServiceError(
            StoryMemoryServiceStage.PERSISTENCE,
            write.error_code or StoryMemoryServiceErrorCode.ATOMIC_WRITE_FAILED,
        )
    return replace(
        result,
        write_status=write.status,
        bytes_written=write.bytes_written,
    )


def resolve_engineering_story_id(
    memory: AuthoritativeEngineeringStoryMemory,
    story_id: str,
    *,
    project_id: str | None = None,
) -> str | None:
    if not isinstance(memory, AuthoritativeEngineeringStoryMemory):
        raise TypeError("memory must be authoritative Story Memory")
    try:
        requested = _canonical_story_id(story_id)
        resolved = resolve_canonical_story_alias(memory.identity_map, requested)
    except (TypeError, ValueError):
        return None
    identity = next(
        (
            item
            for item in memory.identity_map.canonical_identities
            if item.canonical_story_id == resolved
        ),
        None,
    )
    if identity is None:
        return None
    if project_id is not None:
        try:
            requested_project = _exact_project_id(project_id)
        except (TypeError, ValueError):
            return None
        if identity.project_id != requested_project:
            return None
    return resolved


def _history_view(
    history: EngineeringStoryRevisionHistory,
) -> EngineeringStoryView:
    revision = history.current_revision
    lifecycle = revision.lifecycle
    story = replace(
        revision.record.engineering_story,
        story_id=history.canonical_story_id,
        lifecycle=lifecycle,
    )
    provenance = revision.record.provenance
    return EngineeringStoryView(
        canonical_story_id=history.canonical_story_id,
        project_id=history.project_id,
        current_story=story,
        claim_sufficiency=story.claim_sufficiency,
        story_sufficiency=story.story_sufficiency,
        opportunity=story.opportunity,
        lifecycle=lifecycle,
        current_revision_id=revision.revision_id,
        evidence_fact_ids=provenance.evidence_fact_ids,
        capability_fact_ids=provenance.capability_fact_ids,
        claim_boundary_ids=provenance.claim_boundary_ids,
        provenance_fingerprint=provenance.provenance_fingerprint,
        source_lineage_fingerprints=provenance.source_lineage_fingerprints,
    )


def _default_visible(history: EngineeringStoryRevisionHistory) -> bool:
    lifecycle = history.current_revision.lifecycle
    return (
        lifecycle.status is EngineeringStoryStatus.ACTIVE
        and not lifecycle.requires_revalidation
    )


def get_engineering_story_by_id(
    memory: AuthoritativeEngineeringStoryMemory,
    story_id: str,
    *,
    project_id: str | None = None,
    include_non_active: bool = False,
) -> EngineeringStoryView | None:
    resolved = resolve_engineering_story_id(
        memory, story_id, project_id=project_id
    )
    if resolved is None:
        return None
    history = memory.history_for(resolved)
    if history is None or (not include_non_active and not _default_visible(history)):
        return None
    return _history_view(history)


def get_engineering_stories_for_project(
    memory: AuthoritativeEngineeringStoryMemory,
    project_id: str,
    *,
    include_non_active: bool = False,
) -> tuple[EngineeringStoryView, ...]:
    try:
        requested = _exact_project_id(project_id)
    except (TypeError, ValueError):
        return ()
    return tuple(
        _history_view(history)
        for history in memory.histories
        if history.project_id == requested
        and (include_non_active or _default_visible(history))
    )


def get_active_engineering_stories_for_project(
    memory: AuthoritativeEngineeringStoryMemory,
    project_id: str,
) -> tuple[EngineeringStoryView, ...]:
    return get_engineering_stories_for_project(memory, project_id)


def get_current_engineering_story_revision(
    memory: AuthoritativeEngineeringStoryMemory,
    story_id: str,
    *,
    project_id: str | None = None,
    include_non_active: bool = False,
) -> EngineeringStoryRevision | None:
    resolved = resolve_engineering_story_id(
        memory, story_id, project_id=project_id
    )
    if resolved is None:
        return None
    history = memory.history_for(resolved)
    if history is None or (not include_non_active and not _default_visible(history)):
        return None
    return history.current_revision


def get_engineering_story_revision_history(
    memory: AuthoritativeEngineeringStoryMemory,
    story_id: str,
    *,
    project_id: str | None = None,
) -> EngineeringStoryRevisionHistory | None:
    resolved = resolve_engineering_story_id(
        memory, story_id, project_id=project_id
    )
    return None if resolved is None else memory.history_for(resolved)


def get_engineering_story_split_relationships(
    memory: AuthoritativeEngineeringStoryMemory,
    story_id: str,
    *,
    project_id: str | None = None,
) -> tuple[StorySplitRelationship, ...]:
    resolved = resolve_engineering_story_id(
        memory, story_id, project_id=project_id
    )
    if resolved is None:
        return ()
    return tuple(
        item
        for item in memory.identity_map.split_relationships
        if item.parent_canonical_story_id == resolved
        or resolved in item.child_canonical_story_ids
    )


def inspect_authoritative_engineering_story_memory_readiness(
    *,
    path: str | Path | None = None,
    upstream_path: str | Path | None = None,
    compare_upstream: bool = False,
) -> EngineeringStoryMemoryReadiness:
    loaded = load_authoritative_engineering_story_memory(path)
    if loaded.status is StoryMemoryArtifactStatus.MISSING:
        return EngineeringStoryMemoryReadiness(
            StoryMemoryReadinessState.MISSING,
            error_codes=(StoryMemoryServiceErrorCode.ARTIFACT_MISSING.value,),
        )
    if loaded.memory is None:
        return EngineeringStoryMemoryReadiness(
            StoryMemoryReadinessState.INVALID,
            error_codes=((loaded.error_code or StoryMemoryServiceErrorCode.INTEGRITY_FAILED).value,),
        )
    memory = loaded.memory
    counts = _lifecycle_counts(memory)
    requires_revalidation = sum(
        item.current_revision.lifecycle.requires_revalidation
        for item in memory.histories
    )
    stale_source = False
    errors: tuple[str, ...] = ()
    if compare_upstream:
        upstream = load_project_evidence_memory(upstream_path)
        if upstream.status != "ready" or upstream.snapshot is None:
            stale_source = True
            errors = ("upstream_not_ready",)
        else:
            stale_source = (
                upstream.snapshot.content_hash
                != memory.source_artifact.content_hash
            )
            if stale_source:
                errors = ("upstream_content_hash_changed",)
    if counts[EngineeringStoryStatus.CONFLICTED]:
        state = StoryMemoryReadinessState.CONFLICTED
    elif counts[EngineeringStoryStatus.STALE] or requires_revalidation or stale_source:
        state = StoryMemoryReadinessState.STALE_OR_REVALIDATION_REQUIRED
    else:
        state = StoryMemoryReadinessState.READY
    return EngineeringStoryMemoryReadiness(
        state=state,
        schema_version=memory.schema_version,
        logical_fingerprint=memory.logical_fingerprint,
        project_count=len({item.project_id for item in memory.histories}),
        canonical_story_count=len(memory.histories),
        active_story_count=counts[EngineeringStoryStatus.ACTIVE],
        requires_revalidation_count=requires_revalidation,
        stale_story_count=counts[EngineeringStoryStatus.STALE],
        conflicted_story_count=counts[EngineeringStoryStatus.CONFLICTED],
        superseded_story_count=counts[EngineeringStoryStatus.SUPERSEDED],
        revision_count=sum(len(item.revisions) for item in memory.histories),
        alias_count=len(memory.identity_map.aliases),
        split_relationship_count=len(memory.identity_map.split_relationships),
        error_codes=errors,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, refresh, or validate authoritative Engineering Story Memory."
    )
    parser.add_argument("command", choices=("build", "refresh", "validate", "status"))
    parser.add_argument("--upstream", type=Path, default=DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build_and_materialize_authoritative_engineering_story_memory(
                upstream_path=args.upstream,
                output_path=args.output,
            ).to_dict()
        elif args.command == "refresh":
            payload = refresh_and_materialize_authoritative_engineering_story_memory(
                upstream_path=args.upstream,
                output_path=args.output,
            ).to_dict()
        elif args.command == "validate":
            loaded = load_authoritative_engineering_story_memory(args.output)
            payload = inspect_authoritative_engineering_story_memory_readiness(
                path=args.output,
                upstream_path=args.upstream,
                compare_upstream=False,
            ).to_dict()
            if loaded.memory is None:
                raise EngineeringStoryMemoryServiceError(
                    StoryMemoryServiceStage.INTEGRITY,
                    loaded.error_code or StoryMemoryServiceErrorCode.INTEGRITY_FAILED,
                )
        else:
            payload = inspect_authoritative_engineering_story_memory_readiness(
                path=args.output,
                upstream_path=args.upstream,
                compare_upstream=True,
            ).to_dict()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except EngineeringStoryMemoryServiceError as exc:
        print(json.dumps({
            "status": "failed",
            "stage": exc.stage.value,
            "error_code": exc.code.value,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_ENGINEERING_STORY_MEMORY_SCHEMA_VERSION",
    "CANDIDATE_MEMORY_SCHEMA_VERSION",
    "DEFAULT_AUTHORITATIVE_ENGINEERING_STORY_MEMORY_PATH",
    "AuthoritativeEngineeringStoryMemory",
    "AuthoritativeStoryMemoryLoadResult",
    "EngineeringStoryMemoryOperationResult",
    "EngineeringStoryMemoryReadiness",
    "EngineeringStoryMemoryServiceError",
    "EngineeringStoryMemoryValidationResult",
    "EngineeringStoryMemoryWriteResult",
    "EngineeringStorySourceArtifact",
    "EngineeringStoryView",
    "StoryMemoryArtifactStatus",
    "StoryMemoryDiagnosticCode",
    "StoryMemoryOperation",
    "StoryMemoryReadinessState",
    "StoryMemoryServiceErrorCode",
    "StoryMemoryServiceStage",
    "StoryMemoryWriteStatus",
    "build_and_materialize_authoritative_engineering_story_memory",
    "build_authoritative_engineering_story_memory",
    "get_active_engineering_stories_for_project",
    "get_current_engineering_story_revision",
    "get_engineering_stories_for_project",
    "get_engineering_story_by_id",
    "get_engineering_story_revision_history",
    "get_engineering_story_split_relationships",
    "inspect_authoritative_engineering_story_memory_readiness",
    "load_authoritative_engineering_story_memory",
    "main",
    "refresh_and_materialize_authoritative_engineering_story_memory",
    "refresh_authoritative_engineering_story_memory",
    "resolve_engineering_story_id",
    "serialize_authoritative_engineering_story_memory",
    "validate_authoritative_engineering_story_memory",
    "write_authoritative_engineering_story_memory",
]
