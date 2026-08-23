"""Strict bounded models for read-only Chroma operational state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA = "chroma_operational_collection_status.v1"
CHROMA_OPERATIONAL_SERVER_STATES = frozenset({"available", "degraded", "unavailable"})
CHROMA_OPERATIONAL_INTEGRITY_STATES = frozenset(
    {"valid", "collection_missing", "unavailable", "integrity_failure"}
)

_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SAFE_DETAIL_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_SAFE_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$"
)
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:-]{0,200}$")


class ChromaOperationalModelError(ValueError):
    """A strict model failure represented only by a stable safe code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _safe_identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ChromaOperationalModelError(code)
    return value


def _safe_optional_value(value: Any, code: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SAFE_VALUE_RE.fullmatch(value):
        raise ChromaOperationalModelError(code)
    return value or None


@dataclass(frozen=True, slots=True, repr=False)
class ChromaOperationalRepositorySummary:
    repository: str
    project_id: str | None = None
    source_type: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not _SAFE_REPOSITORY_RE.fullmatch(
            self.repository
        ):
            raise ChromaOperationalModelError("invalid_operational_repository")
        object.__setattr__(
            self,
            "project_id",
            _safe_optional_value(self.project_id, "invalid_operational_project_id"),
        )
        object.__setattr__(
            self,
            "source_type",
            _safe_optional_value(self.source_type, "invalid_operational_source_type"),
        )
        object.__setattr__(
            self,
            "updated_at",
            _safe_optional_value(self.updated_at, "invalid_operational_updated_at"),
        )

    def safe_summary(self) -> dict[str, str | None]:
        return {
            "repository": self.repository,
            "project_id": self.project_id,
            "source_type": self.source_type,
            "updated_at": self.updated_at,
        }

    def __repr__(self) -> str:
        return f"ChromaOperationalRepositorySummary(repository={self.repository!r})"


@dataclass(frozen=True, slots=True, repr=False)
class ChromaOperationalCollectionStatus:
    schema: str
    server_state: str
    collection_semantic_id: str
    collection_name: str
    collection_available: bool
    safe_record_count: int | None
    latency_ms: int
    integrity_state: str
    detail: str
    repositories: tuple[ChromaOperationalRepositorySummary, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA:
            raise ChromaOperationalModelError("unsupported_operational_status_schema")
        if self.server_state not in CHROMA_OPERATIONAL_SERVER_STATES:
            raise ChromaOperationalModelError("invalid_operational_server_state")
        object.__setattr__(
            self,
            "collection_semantic_id",
            _safe_identifier(
                self.collection_semantic_id, "invalid_operational_collection_id"
            ),
        )
        object.__setattr__(
            self,
            "collection_name",
            _safe_identifier(self.collection_name, "invalid_operational_collection_name"),
        )
        if not isinstance(self.collection_available, bool):
            raise ChromaOperationalModelError("invalid_operational_collection_availability")
        if self.safe_record_count is not None and (
            not isinstance(self.safe_record_count, int)
            or isinstance(self.safe_record_count, bool)
            or self.safe_record_count < 0
        ):
            raise ChromaOperationalModelError("invalid_operational_record_count")
        if (
            not isinstance(self.latency_ms, int)
            or isinstance(self.latency_ms, bool)
            or not 0 <= self.latency_ms <= 30_000
        ):
            raise ChromaOperationalModelError("invalid_operational_latency")
        if self.integrity_state not in CHROMA_OPERATIONAL_INTEGRITY_STATES:
            raise ChromaOperationalModelError("invalid_operational_integrity_state")
        if not isinstance(self.detail, str) or not _SAFE_DETAIL_RE.fullmatch(self.detail):
            raise ChromaOperationalModelError("invalid_operational_detail")
        if not isinstance(self.repositories, tuple) or any(
            not isinstance(item, ChromaOperationalRepositorySummary)
            for item in self.repositories
        ):
            raise ChromaOperationalModelError("invalid_operational_repositories")
        repository_names = tuple(item.repository for item in self.repositories)
        if repository_names != tuple(sorted(repository_names)) or len(
            repository_names
        ) != len(set(repository_names)):
            raise ChromaOperationalModelError("invalid_operational_repository_order")
        valid = self.integrity_state == "valid"
        if valid != self.collection_available or valid != (
            self.safe_record_count is not None
        ):
            raise ChromaOperationalModelError("inconsistent_operational_collection_state")
        if valid != (self.server_state == "available"):
            raise ChromaOperationalModelError("inconsistent_operational_server_state")
        if not valid and self.repositories:
            raise ChromaOperationalModelError("unsafe_operational_repository_state")

    @property
    def available(self) -> bool:
        return self.integrity_state == "valid"

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "server_state": self.server_state,
            "collection_semantic_id": self.collection_semantic_id,
            "collection_name": self.collection_name,
            "collection_available": self.collection_available,
            "available": self.available,
            "safe_record_count": self.safe_record_count,
            "latency_ms": self.latency_ms,
            "integrity_state": self.integrity_state,
            "detail": self.detail,
            "repositories": [item.safe_summary() for item in self.repositories],
        }

    def __repr__(self) -> str:
        return (
            "ChromaOperationalCollectionStatus("
            f"collection_semantic_id={self.collection_semantic_id!r}, "
            f"server_state={self.server_state!r}, "
            f"integrity_state={self.integrity_state!r}, "
            f"safe_record_count={self.safe_record_count!r})"
        )
