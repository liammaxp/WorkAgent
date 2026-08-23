"""HTTP-only operational status, existence, and safe-count reads for Chroma."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from backend.chroma_collection_registry import (
    GITHUB_EVIDENCE_SEMANTIC_ID,
    UnknownCollectionSemanticId,
    get_collection_definition,
)
from backend.chroma_config import (
    ChromaConfigurationError,
    ChromaDeploymentConfig,
    load_chroma_deployment_config,
)
from backend.chroma_http_client_factory import (
    ChromaAccessLifecycle,
    ChromaCollectionAuthorityMismatch,
    ChromaFactoryDisabled,
    ChromaHttpClientFactory,
)
from backend.chroma_http_transport import (
    ChromaCollectionMissing,
    ChromaTransportProtocolError,
    ChromaTransportResponseError,
    ChromaTransportTimeout,
    ChromaTransportUnavailable,
)
from backend.chroma_operational_models import (
    CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA,
    ChromaOperationalCollectionStatus,
    ChromaOperationalRepositorySummary,
)
from backend.project_repository_identity import (
    normalize_project_id,
    normalize_repository_identity,
)


OPERATIONAL_READER_CONSUMER_ID = "chroma_operational_reader"
OPERATIONAL_PAGE_SIZE = 200
MAX_OPERATIONAL_INVENTORY_RECORDS = 10_000
MAX_OPERATIONAL_LATENCY_MS = 30_000

FactoryBuilder = Callable[[ChromaDeploymentConfig], ChromaHttpClientFactory]
ConfigProvider = Callable[[], ChromaDeploymentConfig]
Clock = Callable[[], float]


def _default_factory_builder(config: ChromaDeploymentConfig) -> ChromaHttpClientFactory:
    return ChromaHttpClientFactory(config)


class ChromaOperationalReader:
    """Fail-closed adapter over the central factory and bounded HTTP transport."""

    __slots__ = ("_clock", "_config_provider", "_factory_builder")

    def __init__(
        self,
        *,
        config_provider: ConfigProvider = load_chroma_deployment_config,
        factory_builder: FactoryBuilder = _default_factory_builder,
        clock: Clock = time.monotonic,
    ) -> None:
        if not callable(config_provider) or not callable(factory_builder) or not callable(clock):
            raise TypeError("invalid_chroma_operational_reader_dependency")
        self._config_provider = config_provider
        self._factory_builder = factory_builder
        self._clock = clock

    def _latency(self, started: float) -> int:
        try:
            value = int(max(0.0, (self._clock() - started) * 1000.0))
        except Exception:
            return 0
        return min(value, MAX_OPERATIONAL_LATENCY_MS)

    def _status(
        self,
        semantic_collection_id: str,
        *,
        server_state: str,
        collection_available: bool,
        count: int | None,
        integrity_state: str,
        detail: str,
        started: float,
        repositories: tuple[ChromaOperationalRepositorySummary, ...] = (),
    ) -> ChromaOperationalCollectionStatus:
        definition = get_collection_definition(semantic_collection_id)
        return ChromaOperationalCollectionStatus(
            schema=CHROMA_OPERATIONAL_COLLECTION_STATUS_SCHEMA,
            server_state=server_state,
            collection_semantic_id=definition.semantic_id,
            collection_name=definition.collection_name,
            collection_available=collection_available,
            safe_record_count=count,
            latency_ms=self._latency(started),
            integrity_state=integrity_state,
            detail=detail,
            repositories=repositories,
        )

    @staticmethod
    def _close_transport(transport: Any) -> None:
        try:
            close = getattr(transport, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    @staticmethod
    def _repository_summary(metadata: Any) -> ChromaOperationalRepositorySummary | None:
        if not isinstance(metadata, Mapping):
            return None
        repository = normalize_repository_identity(metadata.get("repository"))
        if not repository:
            return None
        project_id = normalize_project_id(
            metadata.get("project_id") or metadata.get("repository_project_id")
        )
        source_type = metadata.get("source_type")
        updated_at = metadata.get("updated_at")
        safe_source_type = (
            source_type
            if isinstance(source_type, str)
            and source_type
            and len(source_type) <= 200
            and all(character.isalnum() or character in "_.:-" for character in source_type)
            else None
        )
        return ChromaOperationalRepositorySummary(
            repository=repository,
            project_id=project_id or None,
            source_type=safe_source_type,
            updated_at=(
                updated_at
                if isinstance(updated_at, str)
                and updated_at
                and len(updated_at) <= 200
                and all(
                    character.isalnum() or character in "_.:-+TZ"
                    for character in updated_at
                )
                else None
            ),
        )

    def _repository_inventory(
        self,
        handle: Any,
        count: int,
    ) -> tuple[ChromaOperationalRepositorySummary, ...]:
        if count > MAX_OPERATIONAL_INVENTORY_RECORDS:
            raise ChromaCollectionAuthorityMismatch("operational_inventory_limit_exceeded")
        records_seen = 0
        record_ids: set[str] = set()
        repositories: dict[str, ChromaOperationalRepositorySummary] = {}
        while records_seen < count:
            requested = min(OPERATIONAL_PAGE_SIZE, count - records_seen)
            page = handle.safe_get_page(limit=requested, offset=records_seen)
            if len(page.ids) != requested or page.metadatas is None:
                raise ChromaCollectionAuthorityMismatch("operational_inventory_incomplete")
            for record_id, metadata in zip(page.ids, page.metadatas):
                if record_id in record_ids:
                    raise ChromaCollectionAuthorityMismatch(
                        "operational_inventory_duplicate_record"
                    )
                record_ids.add(record_id)
                summary = self._repository_summary(metadata)
                if summary is not None:
                    existing = repositories.get(summary.repository)
                    if existing is None:
                        repositories[summary.repository] = summary
                    elif existing != summary:
                        if (
                            existing.project_id
                            and summary.project_id
                            and existing.project_id != summary.project_id
                        ):
                            raise ChromaCollectionAuthorityMismatch(
                                "operational_repository_project_conflict"
                            )
                        repositories[summary.repository] = ChromaOperationalRepositorySummary(
                            repository=summary.repository,
                            project_id=existing.project_id or summary.project_id,
                            source_type=None,
                            updated_at=max(
                                existing.updated_at or "",
                                summary.updated_at or "",
                            )
                            or None,
                        )
            records_seen += requested
        if handle.safe_count() != count:
            raise ChromaCollectionAuthorityMismatch("operational_inventory_snapshot_changed")
        return tuple(repositories[key] for key in sorted(repositories))

    def read_collection_status(
        self,
        semantic_collection_id: str,
        *,
        include_repository_inventory: bool = False,
    ) -> ChromaOperationalCollectionStatus:
        """Return one bounded status; all failures are semantic safe states."""

        started = self._clock()
        try:
            definition = get_collection_definition(semantic_collection_id)
        except UnknownCollectionSemanticId:
            raise ValueError("unknown_chroma_operational_collection") from None
        if include_repository_inventory and definition.semantic_id != GITHUB_EVIDENCE_SEMANTIC_ID:
            raise ValueError("unsupported_chroma_operational_inventory")
        transport: Any = None
        try:
            config = self._config_provider()
            if config.is_disabled:
                return self._status(
                    definition.semantic_id,
                    server_state="unavailable",
                    collection_available=False,
                    count=None,
                    integrity_state="unavailable",
                    detail="deployment_disabled",
                    started=started,
                )
            factory = self._factory_builder(config)
            transport = factory.get_transport()
            transport.heartbeat()
            handle = factory.get_collection_handle(
                definition.semantic_id,
                ChromaAccessLifecycle.READ,
                OPERATIONAL_READER_CONSUMER_ID,
                creation_requested=False,
            )
            count = handle.safe_count()
            repositories = (
                self._repository_inventory(handle, count)
                if include_repository_inventory
                else ()
            )
            return self._status(
                definition.semantic_id,
                server_state="available",
                collection_available=True,
                count=count,
                integrity_state="valid",
                detail="ready",
                started=started,
                repositories=repositories,
            )
        except ChromaCollectionMissing:
            return self._status(
                definition.semantic_id,
                server_state="degraded",
                collection_available=False,
                count=None,
                integrity_state="collection_missing",
                detail="collection_missing",
                started=started,
            )
        except ChromaTransportTimeout:
            return self._status(
                definition.semantic_id,
                server_state="unavailable",
                collection_available=False,
                count=None,
                integrity_state="unavailable",
                detail="transport_timeout",
                started=started,
            )
        except (ChromaTransportUnavailable, ChromaFactoryDisabled, ChromaConfigurationError):
            return self._status(
                definition.semantic_id,
                server_state="unavailable",
                collection_available=False,
                count=None,
                integrity_state="unavailable",
                detail="server_unavailable",
                started=started,
            )
        except (
            ChromaCollectionAuthorityMismatch,
            ChromaTransportProtocolError,
            ChromaTransportResponseError,
        ):
            return self._status(
                definition.semantic_id,
                server_state="degraded",
                collection_available=False,
                count=None,
                integrity_state="integrity_failure",
                detail="integrity_failure",
                started=started,
            )
        except Exception:
            return self._status(
                definition.semantic_id,
                server_state="unavailable",
                collection_available=False,
                count=None,
                integrity_state="unavailable",
                detail="operational_read_failed",
                started=started,
            )
        finally:
            self._close_transport(transport)

    def safe_count(self, semantic_collection_id: str) -> int:
        status = self.read_collection_status(semantic_collection_id)
        return status.safe_record_count if status.available else 0
