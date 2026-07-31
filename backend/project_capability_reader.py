"""Default-off, read-only access to authoritative Project Capability Memory.

The reader is deliberately passive: it loads validated local artifacts, checks
their source lineage, and returns only authoritative ``ProjectCapabilityFact``
models.  It never builds, persists, repairs, backfills, or falls back to an
upstream lifecycle representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend import evidence_memory as hash_module
from backend import project_capability_memory as capability_memory_module
from backend import project_evidence_memory as evidence_memory_module
from backend.project_evidence_models import ProjectCapabilityFact


PROJECT_CAPABILITY_MEMORY_FLAG = "USE_PROJECT_CAPABILITY_MEMORY"
ENABLED_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
DISABLED_FLAG_VALUES = frozenset({"", "0", "false", "no", "off"})
PROJECT_CAPABILITY_READER_STATUSES = frozenset({
    "disabled",
    "missing",
    "empty",
    "ready",
    "stale",
    "invalid",
    "error",
})

MAX_READER_DIAGNOSTICS = 32
MAX_READER_CODES = 32
MAX_PROJECT_ID_LENGTH = 300
_SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ALL_PROJECTS = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _codes(values: Sequence[str]) -> tuple[str, ...]:
    safe = {
        value
        for value in values
        if isinstance(value, str) and _SAFE_CODE_RE.fullmatch(value)
    }
    return tuple(sorted(safe)[:MAX_READER_CODES])


def _clone_fact(fact: ProjectCapabilityFact) -> ProjectCapabilityFact:
    if not isinstance(fact, ProjectCapabilityFact):
        raise TypeError("facts must contain ProjectCapabilityFact values")
    return ProjectCapabilityFact.from_dict(fact.to_dict())


def _fact_order(fact: ProjectCapabilityFact) -> tuple[str, str, str, str]:
    return (
        fact.project_id.casefold(),
        fact.project_id,
        fact.capability_type,
        fact.capability_id,
    )


@dataclass(frozen=True)
class ProjectCapabilityReadResult:
    status: str
    enabled: bool
    artifact_schema_version: str | None
    artifact_content_hash: str | None
    source_schema_version: str | None
    source_content_hash: str | None
    source_file_sha256: str | None
    project_count: int
    capability_fact_count: int
    facts: tuple[ProjectCapabilityFact, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in PROJECT_CAPABILITY_READER_STATUSES:
            raise ValueError("unsupported reader status")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a Boolean")
        for name in ("project_count", "capability_fact_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        facts = tuple(sorted((_clone_fact(fact) for fact in self.facts), key=_fact_order))
        if len(facts) != self.capability_fact_count:
            raise ValueError("capability_fact_count must equal returned Fact count")
        if len(self.diagnostics) > MAX_READER_DIAGNOSTICS:
            raise ValueError("reader diagnostics exceed the bounded item count")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "warnings", _codes(self.warnings))
        object.__setattr__(self, "errors", _codes(self.errors))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_safe_dict(self) -> dict[str, Any]:
        """Return bounded status and aggregates without serializing Fact claims."""

        return {
            "status": self.status,
            "enabled": self.enabled,
            "artifact_schema_version": self.artifact_schema_version,
            "artifact_content_hash": self.artifact_content_hash,
            "source_schema_version": self.source_schema_version,
            "source_content_hash": self.source_content_hash,
            "source_file_sha256": self.source_file_sha256,
            "project_count": self.project_count,
            "capability_fact_count": self.capability_fact_count,
            "fact_ids": [fact.capability_id for fact in self.facts],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": _thaw(self.diagnostics),
        }


def _feature_flag_state(feature_enabled: bool | None) -> tuple[bool, bool]:
    if type(feature_enabled) is bool:
        return feature_enabled, True
    if feature_enabled is not None:
        return False, False
    try:
        raw = os.getenv(PROJECT_CAPABILITY_MEMORY_FLAG, "")
    except Exception:
        return False, False
    normalized = raw.strip().casefold() if isinstance(raw, str) else ""
    if normalized in ENABLED_FLAG_VALUES:
        return True, True
    if normalized in DISABLED_FLAG_VALUES:
        return False, True
    return False, False


def is_project_capability_memory_enabled() -> bool:
    """Resolve the semantic feature flag at call time; malformed values fail closed."""

    return _feature_flag_state(None)[0]


def _result(
    status: str,
    *,
    enabled: bool,
    artifact_schema_version: str | None = None,
    artifact_content_hash: str | None = None,
    source_schema_version: str | None = None,
    source_content_hash: str | None = None,
    source_file_sha256: str | None = None,
    project_count: int = 0,
    facts: Sequence[ProjectCapabilityFact] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> ProjectCapabilityReadResult:
    fact_tuple = tuple(facts)
    return ProjectCapabilityReadResult(
        status=status,
        enabled=enabled,
        artifact_schema_version=artifact_schema_version,
        artifact_content_hash=artifact_content_hash,
        source_schema_version=source_schema_version,
        source_content_hash=source_content_hash,
        source_file_sha256=source_file_sha256,
        project_count=project_count,
        capability_fact_count=len(fact_tuple),
        facts=fact_tuple,
        warnings=tuple(warnings),
        errors=tuple(errors),
        diagnostics=diagnostics or {},
    )


def _source_counts(snapshot: Any) -> tuple[int, int, int]:
    return (
        len(snapshot.projects),
        sum(len(project.evidence_facts) for project in snapshot.projects),
        sum(len(project.claim_boundaries) for project in snapshot.projects),
    )


def _lineage_mismatches(
    source_artifact: Any,
    snapshot: Any,
    source_file_sha256: str,
) -> tuple[str, ...]:
    project_count, evidence_count, boundary_count = _source_counts(snapshot)
    comparisons = (
        (source_artifact.schema_version, snapshot.schema_version, "source_schema_version"),
        (source_artifact.content_hash, snapshot.content_hash, "source_content_hash"),
        (source_artifact.project_count, project_count, "source_project_count"),
        (source_artifact.evidence_fact_count, evidence_count, "source_evidence_fact_count"),
        (source_artifact.claim_boundary_count, boundary_count, "source_claim_boundary_count"),
    )
    mismatches = [code for actual, expected, code in comparisons if actual != expected]
    if (
        source_artifact.file_sha256 is not None
        and source_artifact.file_sha256 != source_file_sha256
    ):
        mismatches.append("source_file_sha256")
    return tuple(sorted(mismatches))


def _read_project_capability_memory(
    *,
    project_id: str | object,
    feature_enabled: bool | None,
    capability_memory_path: Path | str,
    evidence_memory_path: Path | str,
) -> ProjectCapabilityReadResult:
    enabled, recognized = _feature_flag_state(feature_enabled)
    base_diagnostics = {
        "feature_flag_enabled": enabled,
        "feature_flag_recognized": recognized,
        "project_lookup_requested": project_id is not _ALL_PROJECTS,
    }
    if not enabled:
        warnings = () if recognized else ("invalid_project_capability_memory_flag",)
        return _result(
            "disabled",
            enabled=False,
            warnings=warnings,
            diagnostics=base_diagnostics,
        )

    if project_id is not _ALL_PROJECTS and (
        not isinstance(project_id, str)
        or not project_id.strip()
        or len(project_id) > MAX_PROJECT_ID_LENGTH
    ):
        return _result(
            "error",
            enabled=True,
            errors=("project_capability_project_id_invalid",),
            diagnostics={**base_diagnostics, "selected_project_known": False},
        )

    try:
        loaded_capability = capability_memory_module.load_project_capability_memory(
            capability_memory_path
        )
    except Exception:
        return _result(
            "error",
            enabled=True,
            errors=("project_capability_reader_error",),
            diagnostics={**base_diagnostics, "capability_loader_status": "error"},
        )
    capability_status = loaded_capability.status
    if capability_status == "missing":
        return _result(
            "missing",
            enabled=True,
            warnings=("project_capability_memory_missing",),
            diagnostics={
                **base_diagnostics,
                "capability_loader_status": "missing",
            },
        )
    if capability_status not in {"empty", "ready"} or loaded_capability.memory is None:
        return _result(
            "invalid",
            enabled=True,
            errors=("project_capability_memory_invalid",),
            diagnostics={
                **base_diagnostics,
                "capability_loader_status": capability_status,
            },
        )

    memory = loaded_capability.memory
    source_artifact = memory.source_artifact
    artifact_diagnostics = {
        **base_diagnostics,
        "artifact_capability_fact_count": len(memory.capability_facts),
        "artifact_project_count": len(memory.projects),
        "capability_loader_status": capability_status,
    }
    try:
        loaded_evidence = evidence_memory_module.load_project_evidence_memory(
            evidence_memory_path
        )
    except Exception:
        return _result(
            "error",
            enabled=True,
            artifact_schema_version=memory.schema_version,
            artifact_content_hash=memory.content_hash,
            source_schema_version=source_artifact.schema_version,
            source_content_hash=source_artifact.content_hash,
            source_file_sha256=source_artifact.file_sha256,
            project_count=len(memory.projects),
            errors=("project_capability_reader_error",),
            diagnostics={**artifact_diagnostics, "evidence_loader_status": "error"},
        )
    if loaded_evidence.status != "ready" or loaded_evidence.snapshot is None:
        evidence_code = (
            "project_evidence_memory_missing"
            if loaded_evidence.status == "missing"
            else "project_evidence_memory_invalid"
        )
        return _result(
            "stale",
            enabled=True,
            artifact_schema_version=memory.schema_version,
            artifact_content_hash=memory.content_hash,
            source_schema_version=source_artifact.schema_version,
            source_content_hash=source_artifact.content_hash,
            source_file_sha256=source_artifact.file_sha256,
            project_count=len(memory.projects),
            warnings=("project_capability_memory_stale", evidence_code),
            diagnostics={
                **artifact_diagnostics,
                "evidence_loader_status": loaded_evidence.status,
                "source_lineage_match": False,
            },
        )

    snapshot = loaded_evidence.snapshot
    try:
        current_source_sha256 = hash_module.stable_hash(
            Path(evidence_memory_path).read_bytes()
        )
    except OSError:
        return _result(
            "stale",
            enabled=True,
            artifact_schema_version=memory.schema_version,
            artifact_content_hash=memory.content_hash,
            source_schema_version=source_artifact.schema_version,
            source_content_hash=source_artifact.content_hash,
            source_file_sha256=source_artifact.file_sha256,
            project_count=len(memory.projects),
            warnings=(
                "project_capability_memory_stale",
                "project_evidence_memory_invalid",
            ),
            diagnostics={
                **artifact_diagnostics,
                "evidence_loader_status": "unreadable",
                "source_lineage_match": False,
            },
        )
    except Exception:
        return _result(
            "error",
            enabled=True,
            artifact_schema_version=memory.schema_version,
            artifact_content_hash=memory.content_hash,
            source_schema_version=source_artifact.schema_version,
            source_content_hash=source_artifact.content_hash,
            source_file_sha256=source_artifact.file_sha256,
            project_count=len(memory.projects),
            errors=("project_capability_reader_error",),
            diagnostics={**artifact_diagnostics, "evidence_loader_status": "error"},
        )

    mismatches = _lineage_mismatches(source_artifact, snapshot, current_source_sha256)
    if mismatches:
        return _result(
            "stale",
            enabled=True,
            artifact_schema_version=memory.schema_version,
            artifact_content_hash=memory.content_hash,
            source_schema_version=source_artifact.schema_version,
            source_content_hash=source_artifact.content_hash,
            source_file_sha256=source_artifact.file_sha256,
            project_count=len(memory.projects),
            warnings=("project_capability_memory_stale",),
            diagnostics={
                **artifact_diagnostics,
                "evidence_loader_status": "ready",
                "lineage_mismatch_fields": mismatches,
                "source_lineage_match": False,
            },
        )

    warnings = tuple(memory.diagnostics.warnings)
    facts: tuple[ProjectCapabilityFact, ...]
    selected_known: bool | None = None
    if project_id is _ALL_PROJECTS:
        facts = tuple(memory.capability_facts)
    else:
        selected_known = any(project.project_id == project_id for project in memory.projects)
        facts = tuple(
            fact for fact in memory.capability_facts if fact.project_id == project_id
        ) if selected_known else ()
        if not selected_known:
            warnings = (*warnings, "project_capability_project_not_found")
        elif not facts:
            warnings = (*warnings, "project_capability_project_has_no_verified_capabilities")

    status = "ready" if facts else "empty"
    final_diagnostics = {
        **artifact_diagnostics,
        "evidence_loader_status": "ready",
        "selected_fact_count": len(facts),
        "source_lineage_match": True,
    }
    if selected_known is not None:
        final_diagnostics["selected_project_known"] = selected_known
    return _result(
        status,
        enabled=True,
        artifact_schema_version=memory.schema_version,
        artifact_content_hash=memory.content_hash,
        source_schema_version=source_artifact.schema_version,
        source_content_hash=source_artifact.content_hash,
        source_file_sha256=source_artifact.file_sha256,
        project_count=len(memory.projects),
        facts=facts,
        warnings=warnings,
        diagnostics=final_diagnostics,
    )


def read_project_capability_memory(
    *,
    feature_enabled: bool | None = None,
    capability_memory_path: Path | str = capability_memory_module.PROJECT_CAPABILITY_MEMORY_PATH,
    evidence_memory_path: Path | str = evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
) -> ProjectCapabilityReadResult:
    """Read all current verified capabilities when the semantic flag is enabled."""

    return _read_project_capability_memory(
        project_id=_ALL_PROJECTS,
        feature_enabled=feature_enabled,
        capability_memory_path=capability_memory_path,
        evidence_memory_path=evidence_memory_path,
    )


def get_verified_project_capabilities(
    project_id: str,
    *,
    feature_enabled: bool | None = None,
    capability_memory_path: Path | str = capability_memory_module.PROJECT_CAPABILITY_MEMORY_PATH,
    evidence_memory_path: Path | str = evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH,
) -> ProjectCapabilityReadResult:
    """Return verified Facts for one exact project ID without fuzzy matching."""

    return _read_project_capability_memory(
        project_id=project_id,
        feature_enabled=feature_enabled,
        capability_memory_path=capability_memory_path,
        evidence_memory_path=evidence_memory_path,
    )


__all__ = [
    "DISABLED_FLAG_VALUES",
    "ENABLED_FLAG_VALUES",
    "PROJECT_CAPABILITY_MEMORY_FLAG",
    "PROJECT_CAPABILITY_READER_STATUSES",
    "ProjectCapabilityReadResult",
    "get_verified_project_capabilities",
    "is_project_capability_memory_enabled",
    "read_project_capability_memory",
]
