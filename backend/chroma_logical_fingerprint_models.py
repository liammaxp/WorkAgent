"""Strict privacy-safe models for logical Chroma collection integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


CHROMA_LOGICAL_FINGERPRINT_SCHEMA = "chroma_logical_fingerprint.v1"
CHROMA_LOGICAL_BASELINE_SCHEMA = "chroma_logical_baseline.v1"
CHROMA_LOGICAL_GATE_SCHEMA = "chroma_logical_integrity_gate.v1"

LOGICAL_INTEGRITY_STATES = frozenset({"valid"})
LOGICAL_COMPARISON_STATES = frozenset(
    {
        "match",
        "record_count_mismatch",
        "record_identity_mismatch",
        "metadata_mismatch",
        "authority_mismatch",
        "collection_missing",
        "schema_mismatch",
        "invalid",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SAFE_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}\.v[1-9][0-9]{0,3}$")
_BACKUP_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")

_FINGERPRINT_FIELDS = frozenset(
    {
        "schema",
        "collection_semantic_id",
        "collection_name",
        "collection_schema_version",
        "record_count",
        "record_id_digest",
        "metadata_digest",
        "authority_digest",
        "aggregate_digest",
        "integrity_state",
    }
)
_BASELINE_FIELDS = frozenset(
    {
        "schema",
        "backup_id",
        "backup_aggregate_sha256",
        "chroma_version",
        "registry_schema",
        "fingerprint_schema",
        "created_at",
        "fingerprints",
        "repeated_run_deterministic",
        "restart_deterministic",
        "workspace_byte_fingerprint_before",
        "workspace_byte_fingerprint_after",
        "workspace_byte_mutated",
        "logical_fingerprints_stable",
        "immutable_backup_unchanged",
        "production_persistence_unchanged",
        "synthetic_validation_passed",
        "privacy_safe",
    }
)


class ChromaLogicalModelError(ValueError):
    """A strict model failure represented by a stable non-sensitive code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ChromaLogicalModelError(code)
    return value


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ChromaLogicalModelError(code)
    return value


def _schema(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SAFE_SCHEMA_RE.fullmatch(value):
        raise ChromaLogicalModelError(code)
    return value


def _nonnegative_int(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChromaLogicalModelError(code)
    return value


def _strict_bool(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        raise ChromaLogicalModelError(code)
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z") or len(value) > 40:
        raise ChromaLogicalModelError("invalid_logical_baseline_timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ChromaLogicalModelError("invalid_logical_baseline_timestamp") from None
    if parsed.utcoffset() is None:
        raise ChromaLogicalModelError("invalid_logical_baseline_timestamp")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class ChromaLogicalFingerprint:
    schema: str
    collection_semantic_id: str
    collection_name: str
    collection_schema_version: str
    record_count: int
    record_id_digest: str
    metadata_digest: str
    authority_digest: str
    aggregate_digest: str
    integrity_state: str

    def __post_init__(self) -> None:
        if self.schema != CHROMA_LOGICAL_FINGERPRINT_SCHEMA:
            raise ChromaLogicalModelError("unsupported_logical_fingerprint_schema")
        object.__setattr__(
            self,
            "collection_semantic_id",
            _identifier(self.collection_semantic_id, "invalid_logical_collection_id"),
        )
        object.__setattr__(
            self,
            "collection_name",
            _identifier(self.collection_name, "invalid_logical_collection_name"),
        )
        object.__setattr__(
            self,
            "collection_schema_version",
            _schema(self.collection_schema_version, "invalid_logical_collection_schema"),
        )
        object.__setattr__(
            self,
            "record_count",
            _nonnegative_int(self.record_count, "invalid_logical_record_count"),
        )
        for field_name in (
            "record_id_digest",
            "metadata_digest",
            "authority_digest",
            "aggregate_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), f"invalid_{field_name}"),
            )
        if self.integrity_state not in LOGICAL_INTEGRITY_STATES:
            raise ChromaLogicalModelError("invalid_logical_integrity_state")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChromaLogicalFingerprint":
        if not isinstance(payload, Mapping) or frozenset(payload) != _FINGERPRINT_FIELDS:
            raise ChromaLogicalModelError("invalid_logical_fingerprint_shape")
        return cls(**{key: payload[key] for key in _FINGERPRINT_FIELDS})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "collection_semantic_id": self.collection_semantic_id,
            "collection_name": self.collection_name,
            "collection_schema_version": self.collection_schema_version,
            "record_count": self.record_count,
            "record_id_digest": self.record_id_digest,
            "metadata_digest": self.metadata_digest,
            "authority_digest": self.authority_digest,
            "aggregate_digest": self.aggregate_digest,
            "integrity_state": self.integrity_state,
        }

    def safe_summary(self) -> dict[str, Any]:
        return self.to_dict()

    def __repr__(self) -> str:
        return (
            "ChromaLogicalFingerprint("
            f"collection_semantic_id={self.collection_semantic_id!r}, "
            f"record_count={self.record_count!r}, "
            f"integrity_state={self.integrity_state!r}, "
            f"aggregate_digest={self.aggregate_digest!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaLogicalComparison:
    state: str
    collection_semantic_id: str
    matches: bool

    def __post_init__(self) -> None:
        if self.state not in LOGICAL_COMPARISON_STATES:
            raise ChromaLogicalModelError("invalid_logical_comparison_state")
        object.__setattr__(
            self,
            "collection_semantic_id",
            _identifier(self.collection_semantic_id, "invalid_logical_collection_id"),
        )
        object.__setattr__(self, "matches", _strict_bool(self.matches, "invalid_logical_match"))
        if self.matches != (self.state == "match"):
            raise ChromaLogicalModelError("inconsistent_logical_comparison")

    def safe_summary(self) -> dict[str, Any]:
        return {
            "collection_semantic_id": self.collection_semantic_id,
            "state": self.state,
            "matches": self.matches,
        }

    def __repr__(self) -> str:
        return (
            "ChromaLogicalComparison("
            f"collection_semantic_id={self.collection_semantic_id!r}, "
            f"state={self.state!r}, matches={self.matches!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaLogicalBaseline:
    schema: str
    backup_id: str
    backup_aggregate_sha256: str
    chroma_version: str
    registry_schema: str
    fingerprint_schema: str
    created_at: str
    fingerprints: tuple[ChromaLogicalFingerprint, ...]
    repeated_run_deterministic: bool
    restart_deterministic: bool
    workspace_byte_fingerprint_before: str
    workspace_byte_fingerprint_after: str
    workspace_byte_mutated: bool
    logical_fingerprints_stable: bool
    immutable_backup_unchanged: bool
    production_persistence_unchanged: bool
    synthetic_validation_passed: bool
    privacy_safe: bool

    def __post_init__(self) -> None:
        if self.schema != CHROMA_LOGICAL_BASELINE_SCHEMA:
            raise ChromaLogicalModelError("unsupported_logical_baseline_schema")
        if not isinstance(self.backup_id, str) or not _BACKUP_ID_RE.fullmatch(self.backup_id):
            raise ChromaLogicalModelError("invalid_logical_baseline_backup_id")
        object.__setattr__(
            self,
            "backup_aggregate_sha256",
            _sha256(self.backup_aggregate_sha256, "invalid_logical_baseline_backup_hash"),
        )
        if not isinstance(self.chroma_version, str) or not _VERSION_RE.fullmatch(self.chroma_version):
            raise ChromaLogicalModelError("invalid_logical_baseline_chroma_version")
        object.__setattr__(
            self, "registry_schema", _schema(self.registry_schema, "invalid_logical_registry_schema")
        )
        if self.fingerprint_schema != CHROMA_LOGICAL_FINGERPRINT_SCHEMA:
            raise ChromaLogicalModelError("invalid_logical_fingerprint_schema_reference")
        object.__setattr__(self, "created_at", _timestamp(self.created_at))
        if not isinstance(self.fingerprints, tuple) or not self.fingerprints:
            raise ChromaLogicalModelError("invalid_logical_baseline_fingerprints")
        if any(not isinstance(item, ChromaLogicalFingerprint) for item in self.fingerprints):
            raise ChromaLogicalModelError("invalid_logical_baseline_fingerprints")
        semantic_ids = tuple(item.collection_semantic_id for item in self.fingerprints)
        if semantic_ids != tuple(sorted(semantic_ids)) or len(semantic_ids) != len(set(semantic_ids)):
            raise ChromaLogicalModelError("invalid_logical_baseline_collection_order")
        for field_name in (
            "workspace_byte_fingerprint_before",
            "workspace_byte_fingerprint_after",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), f"invalid_{field_name}"),
            )
        for field_name in (
            "repeated_run_deterministic",
            "restart_deterministic",
            "workspace_byte_mutated",
            "logical_fingerprints_stable",
            "immutable_backup_unchanged",
            "production_persistence_unchanged",
            "synthetic_validation_passed",
            "privacy_safe",
        ):
            object.__setattr__(
                self,
                field_name,
                _strict_bool(getattr(self, field_name), f"invalid_{field_name}"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChromaLogicalBaseline":
        if not isinstance(payload, Mapping) or frozenset(payload) != _BASELINE_FIELDS:
            raise ChromaLogicalModelError("invalid_logical_baseline_shape")
        raw = payload["fingerprints"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ChromaLogicalModelError("invalid_logical_baseline_fingerprints")
        values = dict(payload)
        values["fingerprints"] = tuple(ChromaLogicalFingerprint.from_mapping(item) for item in raw)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backup_id": self.backup_id,
            "backup_aggregate_sha256": self.backup_aggregate_sha256,
            "chroma_version": self.chroma_version,
            "registry_schema": self.registry_schema,
            "fingerprint_schema": self.fingerprint_schema,
            "created_at": self.created_at,
            "fingerprints": [item.to_dict() for item in self.fingerprints],
            "repeated_run_deterministic": self.repeated_run_deterministic,
            "restart_deterministic": self.restart_deterministic,
            "workspace_byte_fingerprint_before": self.workspace_byte_fingerprint_before,
            "workspace_byte_fingerprint_after": self.workspace_byte_fingerprint_after,
            "workspace_byte_mutated": self.workspace_byte_mutated,
            "logical_fingerprints_stable": self.logical_fingerprints_stable,
            "immutable_backup_unchanged": self.immutable_backup_unchanged,
            "production_persistence_unchanged": self.production_persistence_unchanged,
            "synthetic_validation_passed": self.synthetic_validation_passed,
            "privacy_safe": self.privacy_safe,
        }

    def safe_summary(self) -> dict[str, Any]:
        return self.to_dict()

    def __repr__(self) -> str:
        return (
            "ChromaLogicalBaseline("
            f"backup_id={self.backup_id!r}, collections={len(self.fingerprints)!r}, "
            f"logical_fingerprints_stable={self.logical_fingerprints_stable!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaLogicalIntegrityGate:
    schema: str
    logical_collection_fingerprints_ready: bool
    production_logical_integrity_gate: str
    collection_count: int
    blocker: str | None

    def __post_init__(self) -> None:
        if self.schema != CHROMA_LOGICAL_GATE_SCHEMA:
            raise ChromaLogicalModelError("unsupported_logical_gate_schema")
        object.__setattr__(
            self,
            "logical_collection_fingerprints_ready",
            _strict_bool(
                self.logical_collection_fingerprints_ready,
                "invalid_logical_fingerprint_readiness",
            ),
        )
        if self.production_logical_integrity_gate not in {"logical_integrity_ready", "blocked"}:
            raise ChromaLogicalModelError("invalid_logical_integrity_gate")
        object.__setattr__(
            self,
            "collection_count",
            _nonnegative_int(self.collection_count, "invalid_logical_collection_count"),
        )
        if self.blocker is not None and (
            not isinstance(self.blocker, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", self.blocker)
        ):
            raise ChromaLogicalModelError("invalid_logical_gate_blocker")
        ready = self.logical_collection_fingerprints_ready
        if ready != (self.production_logical_integrity_gate == "logical_integrity_ready"):
            raise ChromaLogicalModelError("inconsistent_logical_integrity_gate")
        if ready != (self.blocker is None):
            raise ChromaLogicalModelError("inconsistent_logical_integrity_blocker")

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_collection_fingerprints_ready": self.logical_collection_fingerprints_ready,
            "production_logical_integrity_gate": self.production_logical_integrity_gate,
            "collection_count": self.collection_count,
            "blocker": self.blocker,
        }

    def __repr__(self) -> str:
        return (
            "ChromaLogicalIntegrityGate("
            f"production_logical_integrity_gate={self.production_logical_integrity_gate!r}, "
            f"collection_count={self.collection_count!r})"
        )
