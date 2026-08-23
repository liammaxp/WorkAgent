"""Bounded immutable models for semantic Chroma mutations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


MAX_WRITE_RECORDS = 100
MAX_WRITE_IDS = 1_000
MAX_WRITE_ID_CHARS = 512
MAX_WRITE_DOCUMENT_CHARS = 512_000
MAX_WRITE_TOTAL_DOCUMENT_CHARS = 1_500_000
MAX_WRITE_METADATA_FIELDS = 128
MAX_WRITE_METADATA_BYTES_PER_RECORD = 32_768
MAX_WRITE_TOTAL_METADATA_BYTES = 1_000_000
MAX_WRITE_EMBEDDING_DIMENSIONS = 8_192
MAX_WRITE_REQUEST_BYTES = 2_000_000


class ChromaWriteModelError(ValueError):
    """Stable model failure that never contains mutation payload data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _safe_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping) or len(metadata) > MAX_WRITE_METADATA_FIELDS:
        raise ChromaWriteModelError("invalid_chroma_write_metadata")
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ChromaWriteModelError("invalid_chroma_write_metadata")
        if value is None or isinstance(value, bool) or isinstance(value, str):
            if isinstance(value, str) and len(value) > MAX_WRITE_DOCUMENT_CHARS:
                raise ChromaWriteModelError("invalid_chroma_write_metadata")
            result[key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
            continue
        if isinstance(value, float) and math.isfinite(value):
            result[key] = float(value)
            continue
        raise ChromaWriteModelError("invalid_chroma_write_metadata")
    try:
        size = len(
            json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
    except (TypeError, ValueError, OverflowError):
        raise ChromaWriteModelError("invalid_chroma_write_metadata") from None
    if size > MAX_WRITE_METADATA_BYTES_PER_RECORD:
        raise ChromaWriteModelError("chroma_write_metadata_too_large")
    return MappingProxyType(result)


def _safe_embedding(embedding: Sequence[float]) -> tuple[float, ...]:
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise ChromaWriteModelError("invalid_chroma_write_embedding")
    if not 1 <= len(embedding) <= MAX_WRITE_EMBEDDING_DIMENSIONS:
        raise ChromaWriteModelError("invalid_chroma_write_embedding")
    values: list[float] = []
    for value in embedding:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ChromaWriteModelError("invalid_chroma_write_embedding")
        values.append(float(value))
    return tuple(values)


@dataclass(frozen=True, slots=True, repr=False)
class ChromaWriteRecord:
    record_id: str
    document: str
    metadata: Mapping[str, Any]
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_id, str)
            or not self.record_id
            or len(self.record_id) > MAX_WRITE_ID_CHARS
        ):
            raise ChromaWriteModelError("invalid_chroma_write_record_id")
        if not isinstance(self.document, str) or len(self.document) > MAX_WRITE_DOCUMENT_CHARS:
            raise ChromaWriteModelError("invalid_chroma_write_document")
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))
        object.__setattr__(self, "embedding", _safe_embedding(self.embedding))

    def safe_summary(self) -> dict[str, int | bool]:
        return {
            "document_present": bool(self.document),
            "metadata_field_count": len(self.metadata),
            "embedding_dimensions": len(self.embedding),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaWriteRecord("
            f"document_present={summary['document_present']!r}, "
            f"metadata_field_count={summary['metadata_field_count']!r}, "
            f"embedding_dimensions={summary['embedding_dimensions']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaWriteResult:
    semantic_collection_id: str
    operation: str
    requested_count: int
    accepted_count: int
    status: str

    def __post_init__(self) -> None:
        if self.operation not in {"upsert", "delete"} or self.status != "applied":
            raise ChromaWriteModelError("invalid_chroma_write_result")
        if (
            not isinstance(self.requested_count, int)
            or isinstance(self.requested_count, bool)
            or self.requested_count < 0
            or not isinstance(self.accepted_count, int)
            or isinstance(self.accepted_count, bool)
            or not 0 <= self.accepted_count <= self.requested_count
        ):
            raise ChromaWriteModelError("invalid_chroma_write_result")

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "operation": self.operation,
            "requested_count": self.requested_count,
            "accepted_count": self.accepted_count,
            "status": self.status,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaWriteResult("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"operation={summary['operation']!r}, "
            f"requested_count={summary['requested_count']!r}, "
            f"accepted_count={summary['accepted_count']!r}, "
            f"status={summary['status']!r})"
        )
