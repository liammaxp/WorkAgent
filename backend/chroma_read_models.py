"""Bounded immutable models for semantic Chroma business reads and vector queries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


MAX_READ_MODEL_RECORDS = 10_000
MAX_READ_MODEL_METADATA_FIELDS = 128
MAX_READ_MODEL_METADATA_STRING_CHARS = 32_768
MAX_READ_MODEL_DOCUMENT_CHARS = 512_000


class ChromaReadModelError(ValueError):
    """Safe model validation failure without record contents."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _safe_record_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ChromaReadModelError("invalid_chroma_read_record_id")
    return value


def _safe_document(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_READ_MODEL_DOCUMENT_CHARS:
        raise ChromaReadModelError("invalid_chroma_read_document")
    return value


def _safe_metadata_value(value: Any) -> str | bool | int | float | None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > MAX_READ_MODEL_METADATA_STRING_CHARS:
            raise ChromaReadModelError("invalid_chroma_read_metadata")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ChromaReadModelError("invalid_chroma_read_metadata")


def project_metadata(
    metadata: Any,
    *,
    allowed_fields: tuple[str, ...],
) -> Mapping[str, str | bool | int | float | None]:
    """Project only explicitly authorized scalar metadata into an immutable mapping."""

    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise ChromaReadModelError("invalid_chroma_read_metadata")
    if (
        not isinstance(allowed_fields, tuple)
        or len(allowed_fields) > MAX_READ_MODEL_METADATA_FIELDS
        or len(allowed_fields) != len(set(allowed_fields))
        or any(not isinstance(key, str) or not key for key in allowed_fields)
    ):
        raise ChromaReadModelError("invalid_chroma_metadata_projection")
    projected = {
        key: _safe_metadata_value(metadata[key])
        for key in allowed_fields
        if key in metadata
    }
    return MappingProxyType(projected)


@dataclass(frozen=True, slots=True, repr=False)
class ChromaReadRecord:
    record_id: str
    document: str | None
    metadata: Mapping[str, str | bool | int | float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _safe_record_id(self.record_id))
        object.__setattr__(self, "document", _safe_document(self.document))
        if not isinstance(self.metadata, Mapping):
            raise ChromaReadModelError("invalid_chroma_read_metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def safe_summary(self) -> dict[str, str | bool | int]:
        return {
            "record_id": self.record_id,
            "content_present": self.document is not None,
            "metadata_field_count": len(self.metadata),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaReadRecord("
            f"record_id={summary['record_id']!r}, "
            f"content_present={summary['content_present']!r}, "
            f"metadata_field_count={summary['metadata_field_count']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaReadResult:
    semantic_collection_id: str
    records: tuple[ChromaReadRecord, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.semantic_collection_id, str)
            or not self.semantic_collection_id
            or not isinstance(self.records, tuple)
            or len(self.records) > MAX_READ_MODEL_RECORDS
            or any(not isinstance(record, ChromaReadRecord) for record in self.records)
        ):
            raise ChromaReadModelError("invalid_chroma_read_result")

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "record_count": len(self.records),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaReadResult("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"record_count={summary['record_count']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaVectorHit:
    record_id: str
    distance: float | None
    document: str | None
    metadata: Mapping[str, str | bool | int | float | None]
    rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _safe_record_id(self.record_id))
        object.__setattr__(self, "document", _safe_document(self.document))
        if self.distance is not None and (
            isinstance(self.distance, bool)
            or not isinstance(self.distance, (int, float))
            or not math.isfinite(float(self.distance))
        ):
            raise ChromaReadModelError("invalid_chroma_vector_distance")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise ChromaReadModelError("invalid_chroma_vector_rank")
        if not isinstance(self.metadata, Mapping):
            raise ChromaReadModelError("invalid_chroma_read_metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def safe_summary(self) -> dict[str, str | bool | int]:
        return {
            "record_id": self.record_id,
            "rank": self.rank,
            "content_present": self.document is not None,
            "metadata_field_count": len(self.metadata),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaVectorHit("
            f"record_id={summary['record_id']!r}, "
            f"rank={summary['rank']!r}, "
            f"content_present={summary['content_present']!r}, "
            f"metadata_field_count={summary['metadata_field_count']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaVectorResult:
    semantic_collection_id: str
    hits: tuple[ChromaVectorHit, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.semantic_collection_id, str)
            or not self.semantic_collection_id
            or not isinstance(self.hits, tuple)
            or len(self.hits) > 100
            or any(not isinstance(hit, ChromaVectorHit) for hit in self.hits)
        ):
            raise ChromaReadModelError("invalid_chroma_vector_result")

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "hit_count": len(self.hits),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaVectorResult("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"hit_count={summary['hit_count']!r})"
        )
