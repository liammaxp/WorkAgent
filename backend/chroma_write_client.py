"""Lazy semantic mutation access through the central Chroma HTTP authority."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from backend.chroma_collection_registry import GITHUB_EVIDENCE_SEMANTIC_ID
from backend.chroma_config import ChromaDeploymentConfig, load_chroma_deployment_config
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaHttpClientFactory,
)
from backend.chroma_write_models import (
    MAX_WRITE_IDS,
    MAX_WRITE_RECORDS,
    MAX_WRITE_REQUEST_BYTES,
    MAX_WRITE_TOTAL_DOCUMENT_CHARS,
    MAX_WRITE_TOTAL_METADATA_BYTES,
    ChromaWriteRecord,
    ChromaWriteResult,
)
from backend.project_repository_identity import (
    authority_to_repository_mapping,
    normalize_project_id,
    normalize_repository_identity,
)


ConfigProvider = Callable[[], ChromaDeploymentConfig]
FactoryBuilder = Callable[[ChromaDeploymentConfig], ChromaHttpClientFactory]


class ChromaSemanticWriteError(RuntimeError):
    """Stable semantic mutation failure without request, content, or response data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ChromaWriteAuthorityViolation(ChromaSemanticWriteError):
    pass


class ChromaWriteLimitExceeded(ChromaSemanticWriteError):
    pass


def _default_factory_builder(config: ChromaDeploymentConfig) -> ChromaHttpClientFactory:
    return ChromaHttpClientFactory(config)


def _safe_ids(ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise ChromaSemanticWriteError("invalid_chroma_delete_request")
    values = tuple(ids)
    if (
        not values
        or len(values) > MAX_WRITE_IDS
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value or len(value) > 512 for value in values)
    ):
        raise ChromaSemanticWriteError("invalid_chroma_delete_request")
    return values


def _safe_records(records: Sequence[ChromaWriteRecord]) -> tuple[ChromaWriteRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ChromaSemanticWriteError("invalid_chroma_upsert_request")
    values = tuple(records)
    if (
        not values
        or len(values) > MAX_WRITE_RECORDS
        or any(not isinstance(record, ChromaWriteRecord) for record in values)
        or len({record.record_id for record in values}) != len(values)
    ):
        raise ChromaSemanticWriteError("invalid_chroma_upsert_request")
    if sum(len(record.document) for record in values) > MAX_WRITE_TOTAL_DOCUMENT_CHARS:
        raise ChromaWriteLimitExceeded("chroma_write_document_limit_exceeded")
    metadata_bytes = 0
    payload_records = []
    for record in values:
        metadata = dict(record.metadata)
        metadata_bytes += len(
            json.dumps(
                metadata,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        payload_records.append(
            {
                "id": record.record_id,
                "document": record.document,
                "metadata": metadata,
                "embedding": list(record.embedding),
            }
        )
    if metadata_bytes > MAX_WRITE_TOTAL_METADATA_BYTES:
        raise ChromaWriteLimitExceeded("chroma_write_metadata_limit_exceeded")
    try:
        request_bytes = len(
            json.dumps(
                payload_records,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError):
        raise ChromaSemanticWriteError("invalid_chroma_upsert_request") from None
    if request_bytes > MAX_WRITE_REQUEST_BYTES:
        raise ChromaWriteLimitExceeded("chroma_write_request_limit_exceeded")
    return values


def _authority_mapping(authority: Any) -> Mapping[str, Any]:
    mapping = authority_to_repository_mapping(authority)
    repository_projects = mapping.get("repository_to_project")
    if (
        not isinstance(repository_projects, Mapping)
        or not repository_projects
        or mapping.get("conflicts")
        or mapping.get("mapping_count") != len(repository_projects)
    ):
        raise ChromaWriteAuthorityViolation("github_write_authority_unavailable")
    for repository, project_id in repository_projects.items():
        if (
            normalize_repository_identity(repository) != repository
            or normalize_project_id(project_id) != project_id
        ):
            raise ChromaWriteAuthorityViolation("github_write_authority_unavailable")
    return mapping


def _validate_github_authority(
    *,
    authority_metadata: Sequence[Mapping[str, Any]] | None,
    repository_authority: Any,
    expected: int,
) -> None:
    if (
        authority_metadata is None
        or isinstance(authority_metadata, (str, bytes))
        or not isinstance(authority_metadata, Sequence)
        or len(authority_metadata) != expected
    ):
        raise ChromaWriteAuthorityViolation("github_write_authority_required")
    mapping = _authority_mapping(repository_authority)
    repository_projects = mapping["repository_to_project"]
    batch_project = ""
    for metadata in authority_metadata:
        if not isinstance(metadata, Mapping):
            raise ChromaWriteAuthorityViolation("github_write_authority_required")
        repository = normalize_repository_identity(metadata.get("repository"))
        project_id = normalize_project_id(metadata.get("project_id"))
        repository_project_id = normalize_project_id(metadata.get("repository_project_id"))
        mapped_project = repository_projects.get(repository)
        if (
            not repository
            or not project_id
            or not repository_project_id
            or project_id != repository_project_id
            or mapped_project != project_id
        ):
            raise ChromaWriteAuthorityViolation("github_write_authority_violation")
        if batch_project and project_id != batch_project:
            raise ChromaWriteAuthorityViolation("github_cross_project_batch_rejected")
        batch_project = project_id


class ChromaWriteClient:
    """Existing-only upsert/delete adapter with no construction-time I/O or retry."""

    __slots__ = ("_config_provider", "_factory_builder")

    def __init__(
        self,
        *,
        config_provider: ConfigProvider = load_chroma_deployment_config,
        factory_builder: FactoryBuilder = _default_factory_builder,
    ) -> None:
        if not callable(config_provider) or not callable(factory_builder):
            raise TypeError("invalid_chroma_write_client_dependency")
        self._config_provider = config_provider
        self._factory_builder = factory_builder

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
            return factory, handle
        except Exception:
            try:
                factory.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _authority(
        semantic_collection_id: str,
        *,
        authority_metadata: Sequence[Mapping[str, Any]] | None,
        repository_authority: Any,
        expected: int,
    ) -> None:
        if semantic_collection_id == GITHUB_EVIDENCE_SEMANTIC_ID:
            _validate_github_authority(
                authority_metadata=authority_metadata,
                repository_authority=repository_authority,
                expected=expected,
            )
        elif authority_metadata is not None or repository_authority is not None:
            raise ChromaWriteAuthorityViolation("unexpected_chroma_write_authority")

    def upsert_records(
        self,
        semantic_collection_id: str,
        *,
        consumer_id: str,
        records: Sequence[ChromaWriteRecord],
        lifecycle: ChromaAccessLifecycle | str = ChromaAccessLifecycle.INDEX,
        authority_metadata: Sequence[Mapping[str, Any]] | None = None,
        repository_authority: Any = None,
    ) -> ChromaWriteResult:
        """Upsert a single bounded stable-ID batch under index authority."""

        if lifecycle not in {ChromaAccessLifecycle.INDEX, ChromaAccessLifecycle.INDEX.value}:
            raise ChromaSemanticWriteError("chroma_index_lifecycle_required")
        safe_records = _safe_records(records)
        self._authority(
            semantic_collection_id,
            authority_metadata=authority_metadata,
            repository_authority=repository_authority,
            expected=len(safe_records),
        )
        factory: Any = None
        try:
            factory, handle = self._open_handle(
                semantic_collection_id,
                ChromaAccessLifecycle.INDEX,
                consumer_id,
            )
            result = handle.safe_upsert_records(
                ids=[record.record_id for record in safe_records],
                embeddings=[record.embedding for record in safe_records],
                documents=[record.document for record in safe_records],
                metadatas=[record.metadata for record in safe_records],
            )
            return ChromaWriteResult(
                semantic_collection_id=semantic_collection_id,
                operation="upsert",
                requested_count=len(safe_records),
                accepted_count=result.affected_count,
                status="applied",
            )
        finally:
            if factory is not None:
                factory.close()

    def delete_records(
        self,
        semantic_collection_id: str,
        *,
        consumer_id: str,
        ids: Sequence[str],
        lifecycle: ChromaAccessLifecycle | str,
        authority_metadata: Sequence[Mapping[str, Any]] | None = None,
        repository_authority: Any = None,
    ) -> ChromaWriteResult:
        """Delete an explicit bounded ID set under write or index authority."""

        try:
            parsed_lifecycle = ChromaAccessLifecycle(lifecycle)
        except (TypeError, ValueError):
            raise ChromaSemanticWriteError("chroma_mutation_lifecycle_required") from None
        if parsed_lifecycle not in {
            ChromaAccessLifecycle.WRITE,
            ChromaAccessLifecycle.INDEX,
        }:
            raise ChromaSemanticWriteError("chroma_mutation_lifecycle_required")
        safe_ids = _safe_ids(ids)
        self._authority(
            semantic_collection_id,
            authority_metadata=authority_metadata,
            repository_authority=repository_authority,
            expected=len(safe_ids),
        )
        factory: Any = None
        try:
            factory, handle = self._open_handle(
                semantic_collection_id,
                parsed_lifecycle,
                consumer_id,
            )
            result = handle.safe_delete_records(ids=safe_ids)
            return ChromaWriteResult(
                semantic_collection_id=semantic_collection_id,
                operation="delete",
                requested_count=len(safe_ids),
                accepted_count=result.affected_count,
                status="applied",
            )
        finally:
            if factory is not None:
                factory.close()
