"""Lazy semantic business-read and vector-query access through central Chroma HTTP."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from backend.chroma_config import ChromaDeploymentConfig, load_chroma_deployment_config
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import MAX_GET_RECORDS, MAX_QUERY_RESULTS
from backend.chroma_read_models import (
    MAX_READ_MODEL_RECORDS,
    ChromaReadRecord,
    ChromaReadResult,
    ChromaVectorHit,
    ChromaVectorResult,
    project_metadata,
)


DEFAULT_READ_PAGE_SIZE = 200
MAX_SEMANTIC_READ_DOCUMENT_CHARS = 2_000_000

ConfigProvider = Callable[[], ChromaDeploymentConfig]
FactoryBuilder = Callable[[ChromaDeploymentConfig], ChromaHttpClientFactory]


class ChromaSemanticReadError(RuntimeError):
    """Safe semantic read failure that never contains documents or raw responses."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ChromaReadLimitExceeded(ChromaSemanticReadError):
    pass


class ChromaReadSnapshotInconsistent(ChromaSemanticReadError):
    pass


def _default_factory_builder(config: ChromaDeploymentConfig) -> ChromaHttpClientFactory:
    return ChromaHttpClientFactory(config)


def _metadata_fields(value: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) > 128
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item or len(item) > 256 for item in value)
    ):
        raise ChromaSemanticReadError("invalid_chroma_metadata_projection")
    return value


class ChromaReadClient:
    """Central-factory semantic adapter with no import-time or construction-time I/O."""

    __slots__ = ("_config_provider", "_factory_builder")

    def __init__(
        self,
        *,
        config_provider: ConfigProvider = load_chroma_deployment_config,
        factory_builder: FactoryBuilder = _default_factory_builder,
    ) -> None:
        if not callable(config_provider) or not callable(factory_builder):
            raise TypeError("invalid_chroma_read_client_dependency")
        self._config_provider = config_provider
        self._factory_builder = factory_builder

    @staticmethod
    def _close_transport(transport: Any) -> None:
        try:
            close = getattr(transport, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def _open_handle(
        self,
        semantic_collection_id: str,
        lifecycle: ChromaAccessLifecycle,
        consumer_id: str,
    ) -> tuple[Any, Any]:
        config = self._config_provider()
        factory = self._factory_builder(config)
        try:
            handle = factory.get_collection_handle(
                semantic_collection_id,
                lifecycle,
                consumer_id,
                creation_requested=False,
            )
            return factory.get_transport(), handle
        except Exception:
            try:
                close = getattr(factory, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise

    @staticmethod
    def _records_from_page(
        page: Any,
        *,
        metadata_fields: tuple[str, ...],
    ) -> tuple[ChromaReadRecord, ...]:
        documents = page.documents if page.documents is not None else (None,) * len(page.ids)
        metadatas = page.metadatas if page.metadatas is not None else (None,) * len(page.ids)
        return tuple(
            ChromaReadRecord(
                record_id=record_id,
                document=document,
                metadata=project_metadata(metadata, allowed_fields=metadata_fields),
            )
            for record_id, document, metadata in zip(page.ids, documents, metadatas)
        )

    def read_records(
        self,
        semantic_collection_id: str,
        *,
        consumer_id: str,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include_documents: bool = False,
        metadata_fields: tuple[str, ...] = (),
        max_records: int = MAX_GET_RECORDS,
        page_size: int = DEFAULT_READ_PAGE_SIZE,
    ) -> ChromaReadResult:
        """Read selected or paginated records with explicit bounds and projection."""

        fields = _metadata_fields(metadata_fields)
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or not 1 <= max_records <= MAX_READ_MODEL_RECORDS
            or not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or not 1 <= page_size <= MAX_GET_RECORDS
            or ids is not None and where is not None
        ):
            raise ChromaSemanticReadError("invalid_chroma_read_request")
        if ids is not None and (
            isinstance(ids, (str, bytes))
            or not isinstance(ids, Sequence)
            or not ids
            or len(ids) > min(max_records, MAX_GET_RECORDS)
        ):
            raise ChromaSemanticReadError("invalid_chroma_read_request")

        transport: Any = None
        try:
            transport, handle = self._open_handle(
                semantic_collection_id,
                ChromaAccessLifecycle.READ,
                consumer_id,
            )
            include_metadata = bool(fields)
            if ids is not None:
                page = handle.safe_get_records(
                    ids=ids,
                    include_documents=include_documents,
                    include_metadata=include_metadata,
                )
                records = self._records_from_page(page, metadata_fields=fields)
                if len({record.record_id for record in records}) != len(records):
                    raise ChromaReadSnapshotInconsistent("duplicate_chroma_read_record")
                return ChromaReadResult(semantic_collection_id, records)

            collection_count = handle.safe_count()
            if where is None and collection_count > max_records:
                raise ChromaReadLimitExceeded("chroma_read_record_limit_exceeded")
            records: list[ChromaReadRecord] = []
            seen: set[str] = set()
            document_chars = 0
            offset = 0
            while len(records) < max_records:
                requested = min(page_size, max_records - len(records))
                page = handle.safe_get_records(
                    where=where,
                    limit=requested,
                    offset=offset,
                    include_documents=include_documents,
                    include_metadata=include_metadata,
                )
                converted = self._records_from_page(page, metadata_fields=fields)
                if any(record.record_id in seen for record in converted):
                    raise ChromaReadSnapshotInconsistent("duplicate_chroma_read_record")
                for record in converted:
                    seen.add(record.record_id)
                    document_chars += len(record.document or "")
                    if document_chars > MAX_SEMANTIC_READ_DOCUMENT_CHARS:
                        raise ChromaReadLimitExceeded("chroma_read_content_limit_exceeded")
                    records.append(record)
                offset += len(converted)
                if len(converted) < requested:
                    break
                if where is None and len(records) == collection_count:
                    break
            if where is not None and len(records) == max_records and collection_count > max_records:
                overflow = handle.safe_get_records(
                    where=where,
                    limit=1,
                    offset=offset,
                    include_documents=False,
                    include_metadata=False,
                )
                if overflow.ids:
                    raise ChromaReadLimitExceeded("chroma_read_record_limit_exceeded")
            if where is None and (
                len(records) != collection_count or handle.safe_count() != collection_count
            ):
                raise ChromaReadSnapshotInconsistent("chroma_read_snapshot_inconsistent")
            return ChromaReadResult(semantic_collection_id, tuple(records))
        finally:
            self._close_transport(transport)

    def vector_query(
        self,
        semantic_collection_id: str,
        *,
        consumer_id: str,
        query_embedding: Sequence[float],
        n_results: int,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include_documents: bool = False,
        metadata_fields: tuple[str, ...] = (),
        include_distances: bool = True,
    ) -> ChromaVectorResult:
        """Run one bounded vector query without exposing stored embeddings."""

        fields = _metadata_fields(metadata_fields)
        if (
            isinstance(query_embedding, (str, bytes))
            or not isinstance(query_embedding, Sequence)
            or not isinstance(n_results, int)
            or isinstance(n_results, bool)
            or not 1 <= n_results <= MAX_QUERY_RESULTS
        ):
            raise ChromaSemanticReadError("invalid_chroma_vector_request")
        transport: Any = None
        try:
            transport, handle = self._open_handle(
                semantic_collection_id,
                ChromaAccessLifecycle.VECTOR_QUERY,
                consumer_id,
            )
            collection_count = handle.safe_count()
            if collection_count == 0:
                return ChromaVectorResult(semantic_collection_id, ())
            limit = min(n_results, collection_count)
            result = handle.safe_vector_query(
                query_embeddings=[query_embedding],
                n_results=limit,
                ids=ids,
                where=where,
                include_documents=include_documents,
                include_metadata=bool(fields),
                include_distances=include_distances,
            )
            record_ids = result.ids[0]
            if len(record_ids) > limit or len(set(record_ids)) != len(record_ids):
                raise ChromaReadSnapshotInconsistent("invalid_chroma_vector_membership")
            documents = result.documents[0] if result.documents is not None else (None,) * len(record_ids)
            metadatas = result.metadatas[0] if result.metadatas is not None else (None,) * len(record_ids)
            distances = result.distances[0] if result.distances is not None else (None,) * len(record_ids)
            hits = tuple(
                ChromaVectorHit(
                    record_id=record_id,
                    distance=distance,
                    document=document,
                    metadata=project_metadata(metadata, allowed_fields=fields),
                    rank=rank,
                )
                for rank, (record_id, distance, document, metadata) in enumerate(
                    zip(record_ids, distances, documents, metadatas),
                    start=1,
                )
            )
            return ChromaVectorResult(semantic_collection_id, hits)
        finally:
            self._close_transport(transport)
