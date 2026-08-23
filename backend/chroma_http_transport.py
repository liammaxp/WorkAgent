"""Bounded WorkAgent-owned transport for the minimal public Chroma HTTP surface."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from backend.chroma_collection_registry import (
    ChromaCollectionDefinition,
    UnknownCollectionSemanticId,
    get_collection_definition,
)
from backend.chroma_config import ChromaDeploymentConfig
from backend.chroma_write_models import (
    MAX_WRITE_DOCUMENT_CHARS,
    MAX_WRITE_EMBEDDING_DIMENSIONS,
    MAX_WRITE_ID_CHARS,
    MAX_WRITE_IDS,
    MAX_WRITE_METADATA_BYTES_PER_RECORD,
    MAX_WRITE_METADATA_FIELDS,
    MAX_WRITE_RECORDS,
    MAX_WRITE_REQUEST_BYTES,
    MAX_WRITE_TOTAL_DOCUMENT_CHARS,
    MAX_WRITE_TOTAL_METADATA_BYTES,
)


CHROMA_PUBLIC_HTTP_API = "v2"
DEFAULT_CHROMA_TENANT = "default_tenant"
DEFAULT_CHROMA_DATABASE = "default_database"
HTTPX_TIMEOUT_DIMENSIONS = ("connect", "read", "write", "pool")
MAX_GET_RECORDS = 1_000
MAX_GET_OFFSET = 100_000
MAX_QUERY_BATCH = 16
MAX_QUERY_RESULTS = 100
MAX_EMBEDDING_DIMENSIONS = 8_192
MAX_FILTER_DEPTH = 8
MAX_FILTER_ITEMS = 256
MAX_METADATA_FIELDS = 128
MAX_PROTOCOL_STRING_LENGTH = 32_768
MAX_DOCUMENT_CHARS = 512_000
MAX_RESPONSE_DOCUMENT_CHARS = 1_500_000
MAX_RESPONSE_BYTES = 2_000_000
_GET_INCLUDED_FIELDS = frozenset({"documents", "metadatas"})
_QUERY_INCLUDED_FIELDS = frozenset({"distances", "documents", "metadatas"})
_GET_RESPONSE_FIELDS = frozenset(
    {"ids", "embeddings", "metadatas", "documents", "uris", "include"}
)
_QUERY_RESPONSE_FIELDS = frozenset(
    {"ids", "embeddings", "metadatas", "documents", "distances", "uris", "include"}
)
_COLLECTION_RESPONSE_FIELDS = frozenset(
    {
        "id",
        "name",
        "configuration_json",
        "tenant",
        "database",
        "log_position",
        "version",
        "metadata",
        "dimension",
        "schema",
    }
)


class ChromaTransportError(RuntimeError):
    """Stable transport failure that never includes URL, payload, or response data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class InvalidChromaTransportConfiguration(ChromaTransportError):
    pass


class ChromaTransportTimeout(ChromaTransportError):
    pass


class ChromaTransportUnavailable(ChromaTransportError):
    pass


class ChromaTransportProtocolError(ChromaTransportError):
    pass


class ChromaTransportResponseError(ChromaTransportError):
    pass


class ChromaCollectionMissing(ChromaTransportError):
    pass


class ChromaTransportClosed(ChromaTransportError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ChromaTransportCount:
    semantic_collection_id: str
    value: int

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "record_count": self.value,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaTransportCount("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"record_count={summary['record_count']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaTransportMutationResult:
    semantic_collection_id: str
    operation: str
    requested_count: int
    affected_count: int

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "operation": self.operation,
            "requested_count": self.requested_count,
            "affected_count": self.affected_count,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaTransportMutationResult("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"operation={summary['operation']!r}, "
            f"requested_count={summary['requested_count']!r}, "
            f"affected_count={summary['affected_count']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaTransportRecords:
    semantic_collection_id: str
    ids: tuple[str, ...]
    metadatas: tuple[Mapping[str, Any] | None, ...] | None
    included: tuple[str, ...]
    documents: tuple[str | None, ...] | None = None

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "record_count": len(self.ids),
            "content_included": self.documents is not None,
            "metadata_included": self.metadatas is not None,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaTransportRecords("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"record_count={summary['record_count']!r}, "
            f"content_included={summary['content_included']!r}, "
            f"metadata_included={summary['metadata_included']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaTransportQueryResult:
    semantic_collection_id: str
    ids: tuple[tuple[str, ...], ...]
    distances: tuple[tuple[float | None, ...], ...] | None
    metadatas: tuple[tuple[Mapping[str, Any] | None, ...], ...] | None
    included: tuple[str, ...]
    documents: tuple[tuple[str | None, ...], ...] | None = None

    def safe_summary(self) -> dict[str, str | int | bool]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "query_count": len(self.ids),
            "result_count": sum(len(batch) for batch in self.ids),
            "distances_included": self.distances is not None,
            "content_included": self.documents is not None,
            "metadata_included": self.metadatas is not None,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaTransportQueryResult("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"query_count={summary['query_count']!r}, "
            f"result_count={summary['result_count']!r}, "
            f"distances_included={summary['distances_included']!r}, "
            f"content_included={summary['content_included']!r}, "
            f"metadata_included={summary['metadata_included']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaTransportCollection:
    semantic_collection_id: str
    collection_name: str
    schema_version: str
    collection_id: str
    tenant: str
    database: str
    dimension: int | None
    _transport: BoundedChromaHttpTransport = field(repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.collection_name

    def count(self) -> ChromaTransportCount:
        return self._transport.count(self)

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str] = ("metadatas",),
    ) -> ChromaTransportRecords:
        return self._transport.get_records(
            self,
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=include,
        )

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str] = ("distances", "metadatas"),
    ) -> ChromaTransportQueryResult:
        return self._transport.query(
            self,
            query_embeddings=query_embeddings,
            n_results=n_results,
            ids=ids,
            where=where,
            include=include,
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> ChromaTransportMutationResult:
        return self._transport.upsert_records(
            self,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete(self, *, ids: Sequence[str]) -> ChromaTransportMutationResult:
        return self._transport.delete_records(self, ids=ids)

    def safe_summary(self) -> dict[str, str | int | None]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "collection_name": self.collection_name,
            "schema_version": self.schema_version,
            "dimension": self.dimension,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaTransportCollection("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"collection_name={summary['collection_name']!r}, "
            f"schema_version={summary['schema_version']!r}, "
            f"dimension={summary['dimension']!r})"
        )


HttpxClientBuilder = Callable[..., Any]


def _default_httpx_client_builder(
    *,
    base_url: str,
    timeout: httpx.Timeout,
    headers: Mapping[str, str],
) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=0, trust_env=False)
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers=dict(headers),
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    )


def _definition(semantic_collection_id: str) -> ChromaCollectionDefinition:
    try:
        return get_collection_definition(semantic_collection_id)
    except UnknownCollectionSemanticId:
        raise ChromaTransportProtocolError("unknown_chroma_collection") from None


def _safe_segment(value: str, code: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ChromaTransportProtocolError(code)
    return quote(value, safe="")


def _validate_ids(ids: Sequence[str] | None) -> list[str] | None:
    if ids is None:
        return None
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_transport_ids")
    values = list(ids)
    if not values or len(values) > MAX_GET_RECORDS:
        raise ChromaTransportProtocolError("invalid_chroma_transport_ids")
    if any(not isinstance(value, str) or not value or len(value) > 512 for value in values):
        raise ChromaTransportProtocolError("invalid_chroma_transport_ids")
    return values


def _request_size(payload: Mapping[str, Any]) -> int:
    try:
        size = len(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_payload") from None
    if size > MAX_WRITE_REQUEST_BYTES:
        raise ChromaTransportProtocolError("chroma_mutation_request_too_large")
    return size


def _write_ids(ids: Sequence[str], *, maximum: int) -> list[str]:
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_ids")
    values = list(ids)
    if (
        not values
        or len(values) > maximum
        or len(values) != len(set(values))
        or any(
            not isinstance(value, str)
            or not value
            or len(value) > MAX_WRITE_ID_CHARS
            for value in values
        )
    ):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_ids")
    return values


def _write_documents(documents: Sequence[str], *, expected: int) -> list[str]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_documents")
    values = list(documents)
    if (
        len(values) != expected
        or any(
            not isinstance(value, str) or len(value) > MAX_WRITE_DOCUMENT_CHARS
            for value in values
        )
        or sum(len(value) for value in values) > MAX_WRITE_TOTAL_DOCUMENT_CHARS
    ):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_documents")
    return values


def _write_embeddings(
    embeddings: Sequence[Sequence[float]], *, expected: int
) -> list[list[float]]:
    if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_embeddings")
    values: list[list[float]] = []
    for embedding in embeddings:
        if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
            raise ChromaTransportProtocolError("invalid_chroma_mutation_embeddings")
        converted: list[float] = []
        for value in embedding:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ChromaTransportProtocolError("invalid_chroma_mutation_embeddings")
            converted.append(float(value))
        if not converted or len(converted) > MAX_WRITE_EMBEDDING_DIMENSIONS:
            raise ChromaTransportProtocolError("invalid_chroma_mutation_embeddings")
        values.append(converted)
    if len(values) != expected:
        raise ChromaTransportProtocolError("invalid_chroma_mutation_embeddings")
    return values


def _write_metadatas(
    metadatas: Sequence[Mapping[str, Any]], *, expected: int
) -> list[dict[str, Any]]:
    if isinstance(metadatas, (str, bytes)) or not isinstance(metadatas, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas")
    if len(metadatas) != expected:
        raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas")
    values: list[dict[str, Any]] = []
    total_bytes = 0
    for metadata in metadatas:
        if not isinstance(metadata, Mapping) or len(metadata) > MAX_WRITE_METADATA_FIELDS:
            raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas")
        converted: dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas")
            if value is None or isinstance(value, (str, bool)):
                converted[key] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                converted[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                converted[key] = float(value)
            else:
                raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas")
        try:
            metadata_bytes = len(
                json.dumps(
                    converted,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        except (TypeError, ValueError, OverflowError):
            raise ChromaTransportProtocolError("invalid_chroma_mutation_metadatas") from None
        if metadata_bytes > MAX_WRITE_METADATA_BYTES_PER_RECORD:
            raise ChromaTransportProtocolError("chroma_mutation_metadata_too_large")
        total_bytes += metadata_bytes
        if total_bytes > MAX_WRITE_TOTAL_METADATA_BYTES:
            raise ChromaTransportProtocolError("chroma_mutation_metadata_too_large")
        values.append(converted)
    return values


def _validate_include(include: Sequence[str], *, query: bool) -> list[str]:
    if isinstance(include, (str, bytes)) or not isinstance(include, Sequence):
        raise ChromaTransportProtocolError("invalid_chroma_transport_include")
    values = list(include)
    allowed = _QUERY_INCLUDED_FIELDS if query else _GET_INCLUDED_FIELDS
    if len(values) != len(set(values)) or any(value not in allowed for value in values):
        raise ChromaTransportProtocolError("unsupported_chroma_transport_include")
    return values


def _json_input(value: Any, *, depth: int = 0, item_budget: list[int] | None = None) -> Any:
    budget = [MAX_FILTER_ITEMS] if item_budget is None else item_budget
    if depth > MAX_FILTER_DEPTH:
        raise ChromaTransportProtocolError("chroma_transport_filter_too_deep")
    budget[0] -= 1
    if budget[0] < 0:
        raise ChromaTransportProtocolError("chroma_transport_filter_too_large")
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > MAX_PROTOCOL_STRING_LENGTH:
            raise ChromaTransportProtocolError("chroma_transport_string_too_large")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChromaTransportProtocolError("invalid_chroma_transport_number")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_FILTER_ITEMS:
            raise ChromaTransportProtocolError("chroma_transport_filter_too_large")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise ChromaTransportProtocolError("invalid_chroma_transport_filter_key")
            result[key] = _json_input(item, depth=depth + 1, item_budget=budget)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > MAX_FILTER_ITEMS:
            raise ChromaTransportProtocolError("chroma_transport_filter_too_large")
        return [_json_input(item, depth=depth + 1, item_budget=budget) for item in value]
    raise ChromaTransportProtocolError("invalid_chroma_transport_json_value")


def _freeze_protocol_value(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_FILTER_DEPTH:
        raise ChromaTransportProtocolError("chroma_transport_response_too_deep")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_PROTOCOL_STRING_LENGTH:
            raise ChromaTransportProtocolError("chroma_transport_response_string_too_large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChromaTransportProtocolError("invalid_chroma_transport_response_number")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_FIELDS:
            raise ChromaTransportProtocolError("chroma_transport_metadata_too_large")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise ChromaTransportProtocolError("invalid_chroma_transport_metadata_key")
            result[key] = _freeze_protocol_value(item, depth=depth + 1)
        return MappingProxyType(result)
    if isinstance(value, list):
        if len(value) > MAX_FILTER_ITEMS:
            raise ChromaTransportProtocolError("chroma_transport_response_too_large")
        return tuple(_freeze_protocol_value(item, depth=depth + 1) for item in value)
    raise ChromaTransportProtocolError("invalid_chroma_transport_response_value")


def _response_ids(value: Any, *, nested: bool) -> tuple[Any, ...]:
    if not isinstance(value, list) or len(value) > MAX_GET_RECORDS:
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_ids")
    if nested:
        batches = []
        for batch in value:
            if not isinstance(batch, list) or len(batch) > MAX_QUERY_RESULTS:
                raise ChromaTransportProtocolError("invalid_chroma_transport_response_ids")
            if any(not isinstance(item, str) or not item for item in batch):
                raise ChromaTransportProtocolError("invalid_chroma_transport_response_ids")
            batches.append(tuple(batch))
        return tuple(batches)
    if any(not isinstance(item, str) or not item for item in value):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_ids")
    return tuple(value)


def _response_included(value: Any, *, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or any(item not in allowed for item in value):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_include")
    if len(value) != len(set(value)):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_include")
    return tuple(value)


def _response_metadatas(value: Any, *, expected: int) -> tuple[Mapping[str, Any] | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != expected:
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_metadatas")
    result = []
    for item in value:
        if item is not None and not isinstance(item, Mapping):
            raise ChromaTransportProtocolError("invalid_chroma_transport_response_metadatas")
        result.append(None if item is None else _freeze_protocol_value(item))
    return tuple(result)


def _response_query_metadatas(
    value: Any,
    *,
    expected_batches: tuple[tuple[str, ...], ...],
) -> tuple[tuple[Mapping[str, Any] | None, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != len(expected_batches):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_metadatas")
    return tuple(
        _response_metadatas(batch, expected=len(ids)) or ()
        for batch, ids in zip(value, expected_batches)
    )


def _response_documents(
    value: Any,
    *,
    expected: int,
    character_budget: list[int] | None = None,
) -> tuple[str | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != expected:
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_documents")
    budget = [MAX_RESPONSE_DOCUMENT_CHARS] if character_budget is None else character_budget
    documents: list[str | None] = []
    for item in value:
        if item is None:
            documents.append(None)
            continue
        if not isinstance(item, str) or len(item) > MAX_DOCUMENT_CHARS:
            raise ChromaTransportProtocolError("unsafe_chroma_document_response")
        budget[0] -= len(item)
        if budget[0] < 0:
            raise ChromaTransportProtocolError("unsafe_chroma_document_response")
        documents.append(item)
    return tuple(documents)


def _response_query_documents(
    value: Any,
    *,
    expected_batches: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str | None, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != len(expected_batches):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_documents")
    budget = [MAX_RESPONSE_DOCUMENT_CHARS]
    return tuple(
        _response_documents(batch, expected=len(ids), character_budget=budget) or ()
        for batch, ids in zip(value, expected_batches)
    )


def _response_distances(
    value: Any,
    *,
    expected_batches: tuple[tuple[str, ...], ...],
) -> tuple[tuple[float | None, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != len(expected_batches):
        raise ChromaTransportProtocolError("invalid_chroma_transport_response_distances")
    result = []
    for batch, ids in zip(value, expected_batches):
        if not isinstance(batch, list) or len(batch) != len(ids):
            raise ChromaTransportProtocolError("invalid_chroma_transport_response_distances")
        converted = []
        for item in batch:
            if item is None:
                converted.append(None)
            elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                converted.append(float(item))
            else:
                raise ChromaTransportProtocolError("invalid_chroma_transport_response_distances")
        result.append(tuple(converted))
    return tuple(result)


class BoundedChromaHttpTransport:
    """One no-retry httpx client with all timeout dimensions derived from config."""

    __slots__ = (
        "_client",
        "_closed",
        "_config",
        "_database",
        "_last_error_category",
        "_tenant",
        "_timeout",
    )

    def __init__(
        self,
        config: ChromaDeploymentConfig,
        *,
        client_builder: HttpxClientBuilder | None = None,
        tenant: str = DEFAULT_CHROMA_TENANT,
        database: str = DEFAULT_CHROMA_DATABASE,
    ) -> None:
        if not isinstance(config, ChromaDeploymentConfig):
            raise InvalidChromaTransportConfiguration("invalid_chroma_transport_configuration")
        if config.is_disabled or not config.uses_http or config.host is None or config.port is None:
            raise InvalidChromaTransportConfiguration("chroma_transport_disabled")
        if not isinstance(config.timeout_seconds, (int, float)) or isinstance(
            config.timeout_seconds, bool
        ):
            raise InvalidChromaTransportConfiguration("invalid_chroma_transport_timeout")
        timeout_seconds = float(config.timeout_seconds)
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 30.0:
            raise InvalidChromaTransportConfiguration("invalid_chroma_transport_timeout")
        self._tenant = _safe_segment(tenant, "invalid_chroma_transport_tenant")
        self._database = _safe_segment(database, "invalid_chroma_transport_database")
        scheme = "https" if config.ssl else "http"
        base_url = f"{scheme}://{config.host}:{config.port}/api/{CHROMA_PUBLIC_HTTP_API}"
        self._timeout = httpx.Timeout(
            timeout_seconds,
            connect=timeout_seconds,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        builder = client_builder or _default_httpx_client_builder
        try:
            client = builder(
                base_url=base_url,
                timeout=self._timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "WorkAgent-Chroma-Transport/1",
                },
            )
        except ChromaTransportError:
            raise
        except Exception:
            raise ChromaTransportUnavailable("chroma_transport_construction_failed") from None
        if client is None or not callable(getattr(client, "request", None)):
            raise ChromaTransportProtocolError("invalid_chroma_http_transport_client")
        self._config = config
        self._client = client
        self._last_error_category = "none"
        self._closed = False

    def get_transport_summary(self) -> dict[str, str | float | bool]:
        return {
            "transport": "http",
            "deployment_mode": self._config.mode.value,
            "timeout_enforced": True,
            "timeout_seconds": self._config.timeout_seconds,
            "timeout_dimensions": "connect,read,write,pool",
            "retry_policy": "none",
            "last_error_category": self._last_error_category,
            "transport_closed": self._closed,
        }

    def __repr__(self) -> str:
        summary = self.get_transport_summary()
        return (
            "BoundedChromaHttpTransport("
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"timeout_enforced={summary['timeout_enforced']!r}, "
            f"timeout_seconds={summary['timeout_seconds']!r}, "
            f"last_error_category={summary['last_error_category']!r}, "
            f"transport_closed={summary['transport_closed']!r})"
        )

    def _record_error(self, category: str) -> None:
        self._last_error_category = category

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        missing_collection_on_404: bool = False,
    ) -> Any:
        if self._closed:
            self._record_error("closed")
            raise ChromaTransportClosed("chroma_transport_closed")
        try:
            response = self._client.request(method, path, json=payload)
        except httpx.ConnectTimeout:
            self._record_error("unavailable")
            raise ChromaTransportUnavailable("chroma_transport_unavailable") from None
        except httpx.TimeoutException:
            self._record_error("timeout")
            raise ChromaTransportTimeout("chroma_transport_timeout") from None
        except (httpx.ConnectError, httpx.NetworkError):
            self._record_error("unavailable")
            raise ChromaTransportUnavailable("chroma_transport_unavailable") from None
        except (httpx.ProtocolError, httpx.DecodingError):
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_protocol_error") from None
        except httpx.RequestError:
            self._record_error("unavailable")
            raise ChromaTransportUnavailable("chroma_transport_unavailable") from None
        except ChromaTransportError:
            raise
        except Exception:
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_request_failed") from None
        status = getattr(response, "status_code", None)
        if status == 404 and missing_collection_on_404:
            self._record_error("missing")
            raise ChromaCollectionMissing("chroma_collection_missing")
        if status in {401, 403}:
            self._record_error("response")
            raise ChromaTransportResponseError("chroma_transport_authority_response_error")
        if isinstance(status, int) and 400 <= status < 500:
            self._record_error("response")
            raise ChromaTransportResponseError("chroma_transport_client_response_error")
        if isinstance(status, int) and status >= 500:
            self._record_error("response")
            raise ChromaTransportResponseError("chroma_transport_server_response_error")
        if not isinstance(status, int) or not 200 <= status < 300:
            self._record_error("response")
            raise ChromaTransportResponseError("chroma_transport_unexpected_status")
        try:
            response_size = len(response.content)
        except Exception:
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_response_unreadable") from None
        if response_size > MAX_RESPONSE_BYTES:
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_response_too_large")
        content_type = str(getattr(response, "headers", {}).get("content-type", ""))
        if "json" not in content_type.casefold():
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_non_json_response")
        try:
            result = response.json()
        except (ValueError, TypeError):
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_malformed_json") from None
        self._last_error_category = "none"
        return result

    @staticmethod
    def _collection_path(collection: ChromaTransportCollection) -> str:
        return (
            f"/tenants/{collection.tenant}/databases/{collection.database}"
            f"/collections/{_safe_segment(collection.collection_id, 'invalid_collection_id')}"
        )

    def _validate_bound_collection(
        self, collection: ChromaTransportCollection
    ) -> ChromaCollectionDefinition:
        if not isinstance(collection, ChromaTransportCollection):
            raise ChromaTransportProtocolError("invalid_chroma_transport_collection")
        definition = _definition(collection.semantic_collection_id)
        if (
            collection.collection_name != definition.collection_name
            or collection.schema_version != definition.schema_version
            or collection._transport is not self
            or collection.tenant != DEFAULT_CHROMA_TENANT
            or collection.database != DEFAULT_CHROMA_DATABASE
        ):
            raise ChromaTransportProtocolError("chroma_collection_authority_mismatch")
        try:
            if str(UUID(collection.collection_id)) != collection.collection_id:
                raise ValueError
        except (ValueError, AttributeError, TypeError):
            raise ChromaTransportProtocolError("invalid_collection_id") from None
        return definition

    def _raise_protocol(self, code: str) -> None:
        self._record_error("protocol")
        raise ChromaTransportProtocolError(code)

    def heartbeat(self) -> int:
        payload = self._request_json("GET", "/heartbeat", operation="heartbeat")
        if not isinstance(payload, Mapping) or set(payload) != {"nanosecond heartbeat"}:
            self._raise_protocol("invalid_chroma_heartbeat_response")
        value = payload["nanosecond heartbeat"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            self._raise_protocol("invalid_chroma_heartbeat_response")
        return value

    def get_collection(self, semantic_collection_id: str) -> ChromaTransportCollection:
        definition = _definition(semantic_collection_id)
        name = _safe_segment(definition.collection_name, "invalid_chroma_collection_name")
        path = f"/tenants/{self._tenant}/databases/{self._database}/collections/{name}"
        payload = self._request_json(
            "GET",
            path,
            operation="collection_lookup",
            missing_collection_on_404=True,
        )
        if not isinstance(payload, Mapping) or not set(payload).issubset(
            _COLLECTION_RESPONSE_FIELDS
        ):
            self._raise_protocol("invalid_chroma_collection_response")
        required = {"id", "name", "tenant", "database", "log_position", "version"}
        if not required.issubset(payload):
            self._raise_protocol("invalid_chroma_collection_response")
        collection_id = payload.get("id")
        try:
            canonical_id = str(UUID(str(collection_id)))
        except (ValueError, AttributeError, TypeError):
            self._raise_protocol("invalid_chroma_collection_response")
        dimension = payload.get("dimension")
        if dimension is not None and (
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        ):
            self._raise_protocol("invalid_chroma_collection_response")
        if (
            payload.get("name") != definition.collection_name
            or payload.get("tenant") != DEFAULT_CHROMA_TENANT
            or payload.get("database") != DEFAULT_CHROMA_DATABASE
        ):
            self._raise_protocol("chroma_collection_authority_mismatch")
        return ChromaTransportCollection(
            semantic_collection_id=definition.semantic_id,
            collection_name=definition.collection_name,
            schema_version=definition.schema_version,
            collection_id=canonical_id,
            tenant=DEFAULT_CHROMA_TENANT,
            database=DEFAULT_CHROMA_DATABASE,
            dimension=dimension,
            _transport=self,
        )

    def count(self, collection: ChromaTransportCollection) -> ChromaTransportCount:
        self._validate_bound_collection(collection)
        payload = self._request_json(
            "GET",
            f"{self._collection_path(collection)}/count",
            operation="count",
            missing_collection_on_404=True,
        )
        if not isinstance(payload, int) or isinstance(payload, bool) or payload < 0:
            self._raise_protocol("invalid_chroma_count_response")
        return ChromaTransportCount(collection.semantic_collection_id, payload)

    def get_records(
        self,
        collection: ChromaTransportCollection,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str] = ("metadatas",),
    ) -> ChromaTransportRecords:
        self._validate_bound_collection(collection)
        safe_ids = _validate_ids(ids)
        if where is not None and not isinstance(where, Mapping):
            raise ChromaTransportProtocolError("invalid_chroma_transport_where")
        safe_where = None if where is None else _json_input(where)
        if safe_ids is None and safe_where is None and limit is None:
            raise ChromaTransportProtocolError("bounded_chroma_get_selector_required")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_GET_RECORDS
        ):
            raise ChromaTransportProtocolError("invalid_chroma_transport_limit")
        if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= MAX_GET_OFFSET:
            raise ChromaTransportProtocolError("invalid_chroma_transport_offset")
        safe_include = _validate_include(include, query=False)
        payload = self._request_json(
            "POST",
            f"{self._collection_path(collection)}/get",
            operation="get",
            payload={
                "ids": safe_ids,
                "where": safe_where,
                "limit": limit,
                "offset": offset,
                "where_document": None,
                "include": safe_include,
            },
            missing_collection_on_404=True,
        )
        if not isinstance(payload, Mapping) or not set(payload).issubset(_GET_RESPONSE_FIELDS):
            self._raise_protocol("invalid_chroma_get_response")
        if any(payload.get(field) is not None for field in ("embeddings", "uris")):
            self._raise_protocol("unsafe_chroma_get_response")
        if payload.get("documents") is not None and "documents" not in safe_include:
            self._raise_protocol("unsafe_chroma_get_response")
        response_ids = _response_ids(payload.get("ids"), nested=False)
        included = _response_included(
            payload.get("include"), allowed=_GET_INCLUDED_FIELDS
        )
        documents = _response_documents(payload.get("documents"), expected=len(response_ids))
        metadatas = _response_metadatas(payload.get("metadatas"), expected=len(response_ids))
        if ("documents" in included) != (documents is not None):
            self._raise_protocol("invalid_chroma_get_response")
        if ("metadatas" in included) != (metadatas is not None):
            self._raise_protocol("invalid_chroma_get_response")
        return ChromaTransportRecords(
            semantic_collection_id=collection.semantic_collection_id,
            ids=response_ids,
            documents=documents,
            metadatas=metadatas,
            included=included,
        )

    def query(
        self,
        collection: ChromaTransportCollection,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str] = ("distances", "metadatas"),
    ) -> ChromaTransportQueryResult:
        self._validate_bound_collection(collection)
        if isinstance(query_embeddings, (str, bytes)) or not isinstance(
            query_embeddings, Sequence
        ):
            raise ChromaTransportProtocolError("invalid_chroma_query_embeddings")
        embeddings = []
        for embedding in query_embeddings:
            if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
                raise ChromaTransportProtocolError("invalid_chroma_query_embeddings")
            values = []
            for value in embedding:
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ChromaTransportProtocolError("invalid_chroma_query_embeddings")
                values.append(float(value))
            if not values or len(values) > MAX_EMBEDDING_DIMENSIONS:
                raise ChromaTransportProtocolError("invalid_chroma_query_embeddings")
            embeddings.append(values)
        if not embeddings or len(embeddings) > MAX_QUERY_BATCH:
            raise ChromaTransportProtocolError("invalid_chroma_query_embeddings")
        if not isinstance(n_results, int) or isinstance(n_results, bool) or not 1 <= n_results <= MAX_QUERY_RESULTS:
            raise ChromaTransportProtocolError("invalid_chroma_query_result_limit")
        safe_ids = _validate_ids(ids)
        if where is not None and not isinstance(where, Mapping):
            raise ChromaTransportProtocolError("invalid_chroma_transport_where")
        safe_where = None if where is None else _json_input(where)
        safe_include = _validate_include(include, query=True)
        payload = self._request_json(
            "POST",
            f"{self._collection_path(collection)}/query",
            operation="query",
            payload={
                "ids": safe_ids,
                "query_embeddings": embeddings,
                "n_results": n_results,
                "where": safe_where,
                "where_document": None,
                "include": safe_include,
            },
            missing_collection_on_404=True,
        )
        if not isinstance(payload, Mapping) or not set(payload).issubset(_QUERY_RESPONSE_FIELDS):
            self._raise_protocol("invalid_chroma_query_response")
        if any(payload.get(field) is not None for field in ("embeddings", "uris")):
            self._raise_protocol("unsafe_chroma_query_response")
        if payload.get("documents") is not None and "documents" not in safe_include:
            self._raise_protocol("unsafe_chroma_query_response")
        response_ids = _response_ids(payload.get("ids"), nested=True)
        if len(response_ids) != len(embeddings):
            self._raise_protocol("invalid_chroma_query_response")
        included = _response_included(payload.get("include"), allowed=_QUERY_INCLUDED_FIELDS)
        distances = _response_distances(payload.get("distances"), expected_batches=response_ids)
        documents = _response_query_documents(
            payload.get("documents"), expected_batches=response_ids
        )
        metadatas = _response_query_metadatas(
            payload.get("metadatas"), expected_batches=response_ids
        )
        if ("distances" in included) != (distances is not None):
            self._raise_protocol("invalid_chroma_query_response")
        if ("documents" in included) != (documents is not None):
            self._raise_protocol("invalid_chroma_query_response")
        if ("metadatas" in included) != (metadatas is not None):
            self._raise_protocol("invalid_chroma_query_response")
        return ChromaTransportQueryResult(
            semantic_collection_id=collection.semantic_collection_id,
            ids=response_ids,
            distances=distances,
            documents=documents,
            metadatas=metadatas,
            included=included,
        )

    def upsert_records(
        self,
        collection: ChromaTransportCollection,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> ChromaTransportMutationResult:
        """Apply one bounded stable-ID upsert without exposing request or response content."""

        self._validate_bound_collection(collection)
        safe_ids = _write_ids(ids, maximum=MAX_WRITE_RECORDS)
        safe_documents = _write_documents(documents, expected=len(safe_ids))
        safe_embeddings = _write_embeddings(embeddings, expected=len(safe_ids))
        if collection.dimension is not None and any(
            len(embedding) != collection.dimension for embedding in safe_embeddings
        ):
            raise ChromaTransportProtocolError("chroma_mutation_embedding_dimension_mismatch")
        safe_metadatas = _write_metadatas(metadatas, expected=len(safe_ids))
        payload = {
            "ids": safe_ids,
            "embeddings": safe_embeddings,
            "documents": safe_documents,
            "metadatas": safe_metadatas,
            "uris": None,
        }
        _request_size(payload)
        response = self._request_json(
            "POST",
            f"{self._collection_path(collection)}/upsert",
            operation="upsert",
            payload=payload,
            missing_collection_on_404=True,
        )
        if response is not None and not (isinstance(response, Mapping) and not response):
            self._raise_protocol("invalid_chroma_upsert_response")
        return ChromaTransportMutationResult(
            semantic_collection_id=collection.semantic_collection_id,
            operation="upsert",
            requested_count=len(safe_ids),
            affected_count=len(safe_ids),
        )

    def delete_records(
        self,
        collection: ChromaTransportCollection,
        *,
        ids: Sequence[str],
    ) -> ChromaTransportMutationResult:
        """Delete one bounded explicit ID set; empty or filter-only deletion is impossible."""

        self._validate_bound_collection(collection)
        safe_ids = _write_ids(ids, maximum=MAX_WRITE_IDS)
        request = {"ids": safe_ids, "where": None, "where_document": None}
        _request_size(request)
        response = self._request_json(
            "POST",
            f"{self._collection_path(collection)}/delete",
            operation="delete",
            payload=request,
            missing_collection_on_404=True,
        )
        if not isinstance(response, Mapping) or set(response) != {"deleted"}:
            self._raise_protocol("invalid_chroma_delete_response")
        deleted = response["deleted"]
        if (
            not isinstance(deleted, int)
            or isinstance(deleted, bool)
            or not 0 <= deleted <= len(safe_ids)
        ):
            self._raise_protocol("invalid_chroma_delete_response")
        return ChromaTransportMutationResult(
            semantic_collection_id=collection.semantic_collection_id,
            operation="delete",
            requested_count=len(safe_ids),
            affected_count=deleted,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._client.close()
        except Exception:
            self._record_error("protocol")
            raise ChromaTransportProtocolError("chroma_transport_close_failed") from None
        self._closed = True
