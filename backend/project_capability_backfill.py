"""Controlled backfill of the accepted Project Capability Memory baseline.

The operation has no path parameters and delegates all lifecycle and persistence
work to the authoritative pipeline and memory modules.  Importing this module is
side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from backend import evidence_memory as hash_module
from backend import project_capability_memory as memory_module
from backend import project_capability_pipeline as pipeline_module
from backend import project_evidence_memory as evidence_memory_module


ROOT_DIR = Path(__file__).parents[1]
AUTHORITATIVE_PROJECT_EVIDENCE_MEMORY_PATH = (
    evidence_memory_module.DEFAULT_PROJECT_EVIDENCE_MEMORY_PATH
)
AUTHORITATIVE_PROJECT_CAPABILITY_MEMORY_PATH = memory_module.PROJECT_CAPABILITY_MEMORY_PATH

ACCEPTED_SOURCE_SCHEMA_VERSION = "project_evidence_memory.v1"
ACCEPTED_SOURCE_CONTENT_HASH = (
    "37967289816ec13638b4b30e31a74f52688acc9bc08ff6c6faf760b2c6180fd3"
)
ACCEPTED_SOURCE_FILE_SHA256 = (
    "95750df456d1fb3dea56cf40891593834a52731414a882896d99aa5a51b3f106"
)
ACCEPTED_SOURCE_PROJECT_COUNT = 11
ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT = 283
ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT = 184
ACCEPTED_SOURCE_CAPABILITY_FACT_COUNT = 0

ACCEPTED_CANDIDATE_COUNT = 57
ACCEPTED_MATCHED_EVIDENCE_COUNT = 196
ACCEPTED_UNMATCHED_EVIDENCE_COUNT = 83
ACCEPTED_AMBIGUOUS_EVIDENCE_COUNT = 4
ACCEPTED_SKIPPED_EVIDENCE_COUNT = 0
ACCEPTED_ASSESSMENT_COUNT = 57
ACCEPTED_ELIGIBLE_ASSESSMENT_COUNT = 0
ACCEPTED_POLICY_COUNT = 0
ACCEPTED_BUILD_RESULT_COUNT = 0
ACCEPTED_CAPABILITY_FACT_COUNT = 0
ACCEPTED_PIPELINE_STATUS = "empty"

PROJECT_CAPABILITY_BACKFILL_STATUSES = frozenset({
    "created",
    "unchanged",
    "source_missing",
    "source_baseline_mismatch",
    "source_invalid",
    "pipeline_failed",
    "persistence_failed",
    "target_invalid",
    "target_conflict",
    "verification_failed",
})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ProjectCapabilityBackfillResult:
    status: str
    source_schema_version: str | None
    source_content_hash: str | None
    source_file_sha256_before: str | None
    source_file_sha256_after: str | None
    target_schema_version: str | None
    target_content_hash: str | None
    target_file_sha256: str | None
    pipeline_status: str | None
    source_project_count: int
    source_evidence_fact_count: int
    source_claim_boundary_count: int
    source_capability_fact_count: int
    candidate_count: int
    assessment_count: int
    eligible_assessment_count: int
    policy_count: int
    build_result_count: int
    capability_fact_count: int
    target_existed_before: bool
    target_written: bool
    target_unchanged: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in PROJECT_CAPABILITY_BACKFILL_STATUSES:
            raise ValueError("unsupported backfill status")
        for name in (
            "source_project_count", "source_evidence_fact_count",
            "source_claim_boundary_count", "source_capability_fact_count",
            "candidate_count", "assessment_count", "eligible_assessment_count",
            "policy_count", "build_result_count", "capability_fact_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        object.__setattr__(self, "errors", tuple(sorted(set(self.errors))))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def to_safe_dict(self) -> dict[str, Any]:
        """Return the bounded aggregate result without paths or lifecycle records."""

        return {
            "status": self.status,
            "source_schema_version": self.source_schema_version,
            "source_content_hash": self.source_content_hash,
            "source_file_sha256_before": self.source_file_sha256_before,
            "source_file_sha256_after": self.source_file_sha256_after,
            "target_schema_version": self.target_schema_version,
            "target_content_hash": self.target_content_hash,
            "target_file_sha256": self.target_file_sha256,
            "pipeline_status": self.pipeline_status,
            "source_project_count": self.source_project_count,
            "source_evidence_fact_count": self.source_evidence_fact_count,
            "source_claim_boundary_count": self.source_claim_boundary_count,
            "source_capability_fact_count": self.source_capability_fact_count,
            "candidate_count": self.candidate_count,
            "assessment_count": self.assessment_count,
            "eligible_assessment_count": self.eligible_assessment_count,
            "policy_count": self.policy_count,
            "build_result_count": self.build_result_count,
            "capability_fact_count": self.capability_fact_count,
            "target_existed_before": self.target_existed_before,
            "target_written": self.target_written,
            "target_unchanged": self.target_unchanged,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "diagnostics": _thaw(self.diagnostics),
        }


def _target_is_git_ignored(target: Path) -> bool:
    try:
        relative = target.resolve(strict=False).relative_to(ROOT_DIR.resolve(strict=False))
    except (OSError, ValueError):
        return False
    try:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative.as_posix()],
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _source_counts(snapshot: Any) -> tuple[int, int, int, int]:
    return (
        len(snapshot.projects),
        sum(len(project.evidence_facts) for project in snapshot.projects),
        sum(len(project.claim_boundaries) for project in snapshot.projects),
        sum(len(project.capability_facts) for project in snapshot.projects),
    )


def _staging_artifact_count(target: Path) -> int:
    try:
        return sum(1 for _item in target.parent.glob(f".{target.name}.*.stage"))
    except OSError:
        return 0


def _make_result(
    status: str,
    *,
    source_schema_version: str | None = None,
    source_content_hash: str | None = None,
    source_file_sha256_before: str | None = None,
    source_file_sha256_after: str | None = None,
    target_schema_version: str | None = None,
    target_content_hash: str | None = None,
    target_file_sha256: str | None = None,
    pipeline_result: Any = None,
    source_counts: tuple[int, int, int, int] = (0, 0, 0, 0),
    target_existed_before: bool = False,
    target_written: bool = False,
    target_unchanged: bool = False,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    staging_artifact_count: int = 0,
) -> ProjectCapabilityBackfillResult:
    project_count, evidence_count, boundary_count, source_capability_count = source_counts
    diagnostics = {
        "ambiguous_evidence_count": getattr(pipeline_result, "ambiguous_evidence_count", 0),
        "matched_evidence_count": getattr(pipeline_result, "matched_evidence_count", 0),
        "projects_with_capabilities": (
            int(pipeline_result.diagnostics.get("projects_with_capabilities", 0))
            if pipeline_result is not None else 0
        ),
        "projects_without_capabilities": (
            int(pipeline_result.diagnostics.get("projects_without_capabilities", 0))
            if pipeline_result is not None else 0
        ),
        "skipped_evidence_count": getattr(pipeline_result, "skipped_evidence_count", 0),
        "staging_artifact_count": staging_artifact_count,
        "unmatched_evidence_count": getattr(pipeline_result, "unmatched_evidence_count", 0),
    }
    return ProjectCapabilityBackfillResult(
        status=status,
        source_schema_version=source_schema_version,
        source_content_hash=source_content_hash,
        source_file_sha256_before=source_file_sha256_before,
        source_file_sha256_after=source_file_sha256_after,
        target_schema_version=target_schema_version,
        target_content_hash=target_content_hash,
        target_file_sha256=target_file_sha256,
        pipeline_status=getattr(pipeline_result, "status", None),
        source_project_count=project_count,
        source_evidence_fact_count=evidence_count,
        source_claim_boundary_count=boundary_count,
        source_capability_fact_count=source_capability_count,
        candidate_count=getattr(pipeline_result, "candidate_count", 0),
        assessment_count=getattr(pipeline_result, "assessment_count", 0),
        eligible_assessment_count=getattr(pipeline_result, "eligible_assessment_count", 0),
        policy_count=getattr(pipeline_result, "policy_count", 0),
        build_result_count=getattr(pipeline_result, "build_result_count", 0),
        capability_fact_count=getattr(pipeline_result, "capability_fact_count", 0),
        target_existed_before=target_existed_before,
        target_written=target_written,
        target_unchanged=target_unchanged,
        warnings=tuple(warnings),
        errors=tuple(errors),
        diagnostics=diagnostics,
    )


def _source_baseline_errors(
    snapshot: Any,
    file_sha256: str,
    counts: tuple[int, int, int, int],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_counts = (
        ACCEPTED_SOURCE_PROJECT_COUNT,
        ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT,
        ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT,
        ACCEPTED_SOURCE_CAPABILITY_FACT_COUNT,
    )
    if snapshot.schema_version != ACCEPTED_SOURCE_SCHEMA_VERSION:
        errors.append("source_schema_mismatch")
    if snapshot.content_hash != ACCEPTED_SOURCE_CONTENT_HASH:
        errors.append("source_content_hash_mismatch")
    if file_sha256 != ACCEPTED_SOURCE_FILE_SHA256:
        errors.append("source_file_sha256_mismatch")
    for actual, expected, code in zip(counts, expected_counts, (
        "source_project_count_mismatch",
        "source_evidence_fact_count_mismatch",
        "source_claim_boundary_count_mismatch",
        "source_capability_fact_count_mismatch",
    )):
        if actual != expected:
            errors.append(code)
    return tuple(errors)


def _pipeline_baseline_errors(result: Any) -> tuple[str, ...]:
    expected = {
        "source_project_count": ACCEPTED_SOURCE_PROJECT_COUNT,
        "source_evidence_fact_count": ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT,
        "source_claim_boundary_count": ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT,
        "source_capability_fact_count": ACCEPTED_SOURCE_CAPABILITY_FACT_COUNT,
        "candidate_count": ACCEPTED_CANDIDATE_COUNT,
        "matched_evidence_count": ACCEPTED_MATCHED_EVIDENCE_COUNT,
        "unmatched_evidence_count": ACCEPTED_UNMATCHED_EVIDENCE_COUNT,
        "ambiguous_evidence_count": ACCEPTED_AMBIGUOUS_EVIDENCE_COUNT,
        "skipped_evidence_count": ACCEPTED_SKIPPED_EVIDENCE_COUNT,
        "assessment_count": ACCEPTED_ASSESSMENT_COUNT,
        "eligible_assessment_count": ACCEPTED_ELIGIBLE_ASSESSMENT_COUNT,
        "policy_count": ACCEPTED_POLICY_COUNT,
        "build_result_count": ACCEPTED_BUILD_RESULT_COUNT,
        "capability_fact_count": ACCEPTED_CAPABILITY_FACT_COUNT,
    }
    errors = [
        f"pipeline_{name}_mismatch"
        for name, value in expected.items()
        if getattr(result, name, None) != value
    ]
    if result.status != ACCEPTED_PIPELINE_STATUS:
        errors.append("pipeline_status_mismatch")
    if result.source_schema_version != ACCEPTED_SOURCE_SCHEMA_VERSION:
        errors.append("pipeline_source_schema_mismatch")
    if result.source_content_hash != ACCEPTED_SOURCE_CONTENT_HASH:
        errors.append("pipeline_source_content_hash_mismatch")
    if result.source_file_sha256 != ACCEPTED_SOURCE_FILE_SHA256:
        errors.append("pipeline_source_file_sha256_mismatch")
    if result.memory is None:
        errors.append("pipeline_memory_missing")
    return tuple(errors)


def _memory_matches_pipeline(memory: Any, pipeline_result: Any) -> bool:
    if memory != pipeline_result.memory or memory.content_hash != pipeline_result.memory.content_hash:
        return False
    source = memory.source_artifact
    return all((
        source.schema_version == ACCEPTED_SOURCE_SCHEMA_VERSION,
        source.content_hash == ACCEPTED_SOURCE_CONTENT_HASH,
        source.file_sha256 == ACCEPTED_SOURCE_FILE_SHA256,
        source.project_count == ACCEPTED_SOURCE_PROJECT_COUNT,
        source.evidence_fact_count == ACCEPTED_SOURCE_EVIDENCE_FACT_COUNT,
        source.claim_boundary_count == ACCEPTED_SOURCE_CLAIM_BOUNDARY_COUNT,
        len(memory.projects) == ACCEPTED_SOURCE_PROJECT_COUNT,
        len(memory.capability_facts) == ACCEPTED_CAPABILITY_FACT_COUNT,
        memory.diagnostics.projects_with_capabilities == 0,
        memory.diagnostics.projects_without_capabilities == ACCEPTED_SOURCE_PROJECT_COUNT,
    ))


def _safe_remove_new_target(target: Path, expected_sha256: str) -> bool:
    try:
        if target.is_file() and hash_module.stable_hash(target.read_bytes()) == expected_sha256:
            target.unlink()
            return not target.exists()
    except OSError:
        pass
    return False


def run_authoritative_project_capability_backfill() -> ProjectCapabilityBackfillResult:
    """Create or confirm the one accepted local capability-memory artifact."""

    source = AUTHORITATIVE_PROJECT_EVIDENCE_MEMORY_PATH
    target = AUTHORITATIVE_PROJECT_CAPABILITY_MEMORY_PATH
    target_existed_before = target.exists()
    if not _target_is_git_ignored(target):
        return _make_result(
            "verification_failed",
            target_existed_before=target_existed_before,
            errors=("target_not_git_ignored",),
        )
    if not source.exists():
        return _make_result(
            "source_missing",
            target_existed_before=target_existed_before,
            errors=("source_artifact_missing",),
        )
    try:
        source_bytes_before = source.read_bytes()
        source_sha_before = hash_module.stable_hash(source_bytes_before)
    except OSError:
        return _make_result(
            "source_invalid",
            target_existed_before=target_existed_before,
            errors=("source_artifact_unreadable",),
        )

    loaded_source = evidence_memory_module.load_project_evidence_memory(source)
    if loaded_source.status != "ready" or loaded_source.snapshot is None:
        status = "source_missing" if loaded_source.status == "missing" else "source_invalid"
        error = "source_artifact_missing" if status == "source_missing" else "source_artifact_invalid"
        return _make_result(
            status,
            source_file_sha256_before=source_sha_before,
            target_existed_before=target_existed_before,
            errors=(error,),
        )
    snapshot = loaded_source.snapshot
    counts = _source_counts(snapshot)
    common = {
        "source_schema_version": snapshot.schema_version,
        "source_content_hash": snapshot.content_hash,
        "source_file_sha256_before": source_sha_before,
        "source_counts": counts,
        "target_existed_before": target_existed_before,
    }
    baseline_errors = _source_baseline_errors(snapshot, source_sha_before, counts)
    if baseline_errors:
        return _make_result("source_baseline_mismatch", errors=baseline_errors, **common)

    try:
        pipeline_result = pipeline_module.run_project_capability_pipeline(
            source_path=source,
            persist=False,
        )
    except Exception:
        return _make_result(
            "pipeline_failed", errors=("pipeline_invocation_failed",), **common
        )
    pipeline_errors = _pipeline_baseline_errors(pipeline_result)
    if pipeline_errors:
        return _make_result(
            "pipeline_failed",
            pipeline_result=pipeline_result,
            errors=pipeline_errors,
            **common,
        )
    if not memory_module.validate_project_capability_memory(pipeline_result.memory).valid:
        return _make_result(
            "pipeline_failed",
            pipeline_result=pipeline_result,
            errors=("pipeline_memory_invalid",),
            **common,
        )
    if not _memory_matches_pipeline(pipeline_result.memory, pipeline_result):
        return _make_result(
            "pipeline_failed",
            pipeline_result=pipeline_result,
            errors=("pipeline_memory_lineage_mismatch",),
            **common,
        )

    try:
        source_bytes_before_persist = source.read_bytes()
        source_sha_before_persist = hash_module.stable_hash(source_bytes_before_persist)
    except OSError:
        source_bytes_before_persist = b""
        source_sha_before_persist = ""
    if (
        source_bytes_before_persist != source_bytes_before
        or source_sha_before_persist != source_sha_before
    ):
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            source_file_sha256_after=source_sha_before_persist or None,
            errors=("source_changed_before_persistence",),
            **common,
        )

    if target.exists():
        loaded_target = memory_module.load_project_capability_memory(target)
        if loaded_target.status not in {"ready", "empty"} or loaded_target.memory is None:
            return _make_result(
                "target_invalid",
                pipeline_result=pipeline_result,
                errors=("existing_target_invalid",),
                **common,
            )
        if not _memory_matches_pipeline(loaded_target.memory, pipeline_result):
            return _make_result(
                "target_conflict",
                pipeline_result=pipeline_result,
                target_schema_version=loaded_target.memory.schema_version,
                target_content_hash=loaded_target.memory.content_hash,
                errors=("existing_target_conflict",),
                **common,
            )
        try:
            target_bytes = target.read_bytes()
            expected_target_bytes = memory_module.serialize_project_capability_memory(
                pipeline_result.memory
            )
            target_sha = hash_module.stable_hash(target_bytes)
            source_bytes_after = source.read_bytes()
            source_sha_after = hash_module.stable_hash(source_bytes_after)
        except OSError:
            return _make_result(
                "verification_failed",
                pipeline_result=pipeline_result,
                errors=("artifact_verification_read_failed",),
                **common,
            )
        if target_bytes != expected_target_bytes:
            return _make_result(
                "target_conflict",
                pipeline_result=pipeline_result,
                source_file_sha256_after=source_sha_after,
                target_schema_version=loaded_target.memory.schema_version,
                target_content_hash=loaded_target.memory.content_hash,
                target_file_sha256=target_sha,
                errors=("existing_target_bytes_conflict",),
                **common,
            )
        staging_count = _staging_artifact_count(target)
        if source_bytes_after != source_bytes_before or source_sha_after != source_sha_before:
            return _make_result(
                "verification_failed",
                pipeline_result=pipeline_result,
                source_file_sha256_after=source_sha_after,
                target_schema_version=loaded_target.memory.schema_version,
                target_content_hash=loaded_target.memory.content_hash,
                target_file_sha256=target_sha,
                errors=("source_changed_after_verification",),
                staging_artifact_count=staging_count,
                **common,
            )
        if staging_count:
            return _make_result(
                "verification_failed",
                pipeline_result=pipeline_result,
                source_file_sha256_after=source_sha_after,
                target_schema_version=loaded_target.memory.schema_version,
                target_content_hash=loaded_target.memory.content_hash,
                target_file_sha256=target_sha,
                errors=("staging_artifact_remaining",),
                staging_artifact_count=staging_count,
                **common,
            )
        return _make_result(
            "unchanged",
            pipeline_result=pipeline_result,
            source_file_sha256_after=source_sha_after,
            target_schema_version=loaded_target.memory.schema_version,
            target_content_hash=loaded_target.memory.content_hash,
            target_file_sha256=target_sha,
            target_unchanged=True,
            warnings=pipeline_result.warnings,
            **common,
        )

    persistence = memory_module.persist_project_capability_memory(pipeline_result.memory, target)
    if persistence.status not in {"created", "unchanged"} or not persistence.round_trip_validated:
        return _make_result(
            "persistence_failed",
            pipeline_result=pipeline_result,
            errors=("capability_memory_persistence_failed",),
            **common,
        )
    loaded_target = memory_module.load_project_capability_memory(target)
    if loaded_target.status not in {"ready", "empty"} or loaded_target.memory is None:
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            target_written=persistence.status == "created",
            errors=("persisted_target_invalid",),
            **common,
        )
    if not _memory_matches_pipeline(loaded_target.memory, pipeline_result):
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            target_written=persistence.status == "created",
            target_schema_version=loaded_target.memory.schema_version,
            target_content_hash=loaded_target.memory.content_hash,
            errors=("persisted_target_mismatch",),
            **common,
        )
    try:
        target_bytes = target.read_bytes()
        expected_target_bytes = memory_module.serialize_project_capability_memory(
            pipeline_result.memory
        )
        target_sha = hash_module.stable_hash(target_bytes)
        source_bytes_after = source.read_bytes()
        source_sha_after = hash_module.stable_hash(source_bytes_after)
    except OSError:
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            target_written=persistence.status == "created",
            errors=("artifact_verification_read_failed",),
            **common,
        )
    if target_bytes != expected_target_bytes:
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            source_file_sha256_after=source_sha_after,
            target_schema_version=loaded_target.memory.schema_version,
            target_content_hash=loaded_target.memory.content_hash,
            target_file_sha256=target_sha,
            target_written=persistence.status == "created",
            errors=("persisted_target_bytes_mismatch",),
            **common,
        )
    staging_count = _staging_artifact_count(target)
    if source_bytes_after != source_bytes_before or source_sha_after != source_sha_before:
        removed = False
        if not target_existed_before and persistence.status == "created":
            removed = _safe_remove_new_target(target, target_sha)
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            source_file_sha256_after=source_sha_after,
            target_schema_version=loaded_target.memory.schema_version,
            target_content_hash=loaded_target.memory.content_hash,
            target_file_sha256=target_sha,
            target_written=persistence.status == "created",
            warnings=("new_target_removed_after_source_change",) if removed else (),
            errors=("source_changed_after_persistence",),
            staging_artifact_count=staging_count,
            **common,
        )
    if staging_count:
        return _make_result(
            "verification_failed",
            pipeline_result=pipeline_result,
            source_file_sha256_after=source_sha_after,
            target_schema_version=loaded_target.memory.schema_version,
            target_content_hash=loaded_target.memory.content_hash,
            target_file_sha256=target_sha,
            target_written=persistence.status == "created",
            errors=("staging_artifact_remaining",),
            staging_artifact_count=staging_count,
            **common,
        )
    final_status = "created" if persistence.status == "created" else "unchanged"
    return _make_result(
        final_status,
        pipeline_result=pipeline_result,
        source_file_sha256_after=source_sha_after,
        target_schema_version=loaded_target.memory.schema_version,
        target_content_hash=loaded_target.memory.content_hash,
        target_file_sha256=target_sha,
        target_written=persistence.status == "created",
        target_unchanged=persistence.status == "unchanged",
        warnings=pipeline_result.warnings,
        **common,
    )


__all__ = [
    "ACCEPTED_SOURCE_CONTENT_HASH",
    "ACCEPTED_SOURCE_FILE_SHA256",
    "AUTHORITATIVE_PROJECT_CAPABILITY_MEMORY_PATH",
    "AUTHORITATIVE_PROJECT_EVIDENCE_MEMORY_PATH",
    "PROJECT_CAPABILITY_BACKFILL_STATUSES",
    "ProjectCapabilityBackfillResult",
    "run_authoritative_project_capability_backfill",
]
