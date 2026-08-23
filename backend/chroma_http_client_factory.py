"""Lazy, fail-closed HTTP client and semantic collection-access authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.chroma_collection_registry import (
    LIFECYCLE_ORDER,
    ChromaCollectionDefinition,
    LegacyCollectionConsumer,
    UnknownCollectionSemanticId,
    get_collection_definition,
    list_registered_collections,
)
from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_http_transport import (
    BoundedChromaHttpTransport,
    ChromaCollectionMissing,
    ChromaTransportMutationResult,
    ChromaTransportQueryResult,
    ChromaTransportRecords,
    ChromaTransportError,
    ChromaTransportProtocolError,
    ChromaTransportUnavailable,
)


class ChromaFactoryState(str, Enum):
    DISABLED = "disabled"
    UNINITIALIZED = "uninitialized"
    READY = "ready"


class ChromaAccessLifecycle(str, Enum):
    READ = "read"
    VECTOR_QUERY = "vector_query"
    WRITE = "write"
    INDEX = "index"
    MIGRATION = "migration"
    MAINTENANCE = "maintenance"
    TEST_ONLY = "test_only"


class ChromaConsumerAuthority(str, Enum):
    APPROVED = "approved"
    LEGACY_MIGRATION_TARGET = "legacy_migration_target"
    TEST_ONLY = "test_only"


if tuple(item.value for item in ChromaAccessLifecycle) != LIFECYCLE_ORDER:
    raise RuntimeError("chroma_access_lifecycle_registry_mismatch")


_ACCESS_MODE_BY_LIFECYCLE = {
    ChromaAccessLifecycle.READ: "read_only",
    ChromaAccessLifecycle.VECTOR_QUERY: "read_only",
    ChromaAccessLifecycle.WRITE: "controlled_write",
    ChromaAccessLifecycle.INDEX: "controlled_index",
    ChromaAccessLifecycle.MIGRATION: "controlled_migration",
    ChromaAccessLifecycle.MAINTENANCE: "controlled_maintenance",
    ChromaAccessLifecycle.TEST_ONLY: "test_only",
}
_UNAVAILABLE_TRANSPORT_ERROR_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "NetworkError",
        "PoolTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteTimeout",
    }
)


class ChromaHttpClientFactoryError(RuntimeError):
    """Stable factory failure that never includes transport or user data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class InvalidChromaFactoryConfiguration(ChromaHttpClientFactoryError):
    pass


class ChromaFactoryDisabled(ChromaHttpClientFactoryError):
    pass


class UnsupportedChromaLifecycle(ChromaHttpClientFactoryError):
    pass


class UnknownChromaCollection(ChromaHttpClientFactoryError):
    pass


class UnknownChromaConsumer(ChromaHttpClientFactoryError):
    pass


class ChromaConsumerNotApproved(ChromaHttpClientFactoryError):
    pass


class ChromaLegacyConsumerRequiresMigration(ChromaHttpClientFactoryError):
    pass


class ChromaCollectionCreationNotAllowed(ChromaHttpClientFactoryError):
    pass


class ChromaCollectionAuthorityMismatch(ChromaHttpClientFactoryError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class ValidatedChromaCollectionAccess:
    semantic_collection_id: str
    collection_name: str
    schema_version: str
    requested_lifecycle: ChromaAccessLifecycle
    consumer_id: str
    consumer_authority: ChromaConsumerAuthority
    creation_allowed: bool
    access_mode: str
    access_permitted: bool
    migration_target: str | None

    def safe_summary(self) -> dict[str, str | bool | None]:
        return {
            "semantic_collection_id": self.semantic_collection_id,
            "collection_name": self.collection_name,
            "schema_version": self.schema_version,
            "requested_lifecycle": self.requested_lifecycle.value,
            "consumer_id": self.consumer_id,
            "consumer_authority": self.consumer_authority.value,
            "creation_allowed": self.creation_allowed,
            "access_mode": self.access_mode,
            "access_permitted": self.access_permitted,
            "migration_target": self.migration_target,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ValidatedChromaCollectionAccess("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"requested_lifecycle={summary['requested_lifecycle']!r}, "
            f"consumer_id={summary['consumer_id']!r}, "
            f"consumer_authority={summary['consumer_authority']!r}, "
            f"creation_allowed={summary['creation_allowed']!r}, "
            f"access_permitted={summary['access_permitted']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaCollectionHandle:
    """Validated semantic handle with an intentionally private collection accessor."""

    access: ValidatedChromaCollectionAccess
    _collection_accessor: Any = field(repr=False, compare=False)

    @property
    def semantic_collection_id(self) -> str:
        return self.access.semantic_collection_id

    @property
    def collection_name(self) -> str:
        return self.access.collection_name

    @property
    def schema_version(self) -> str:
        return self.access.schema_version

    @property
    def requested_lifecycle(self) -> ChromaAccessLifecycle:
        return self.access.requested_lifecycle

    @property
    def consumer_id(self) -> str:
        return self.access.consumer_id

    @property
    def creation_allowed(self) -> bool:
        return False

    @property
    def collection_bound(self) -> bool:
        return self._collection_accessor is not None

    def safe_summary(self) -> dict[str, str | bool | None]:
        return {**self.access.safe_summary(), "collection_bound": self.collection_bound}

    def safe_count(self) -> int:
        """Return only a bounded collection cardinality from the validated accessor."""

        result = self._collection_accessor.count()
        value = getattr(result, "value", result)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_collection_count")
        return value

    def safe_get_page(self, *, limit: int, offset: int) -> ChromaTransportRecords:
        """Return one bounded IDs/allowlisted-metadata candidate page only."""

        result = self._collection_accessor.get(
            limit=limit,
            offset=offset,
            include=("metadatas",),
        )
        if (
            not isinstance(result, ChromaTransportRecords)
            or result.semantic_collection_id != self.semantic_collection_id
            or result.included != ("metadatas",)
            or result.metadatas is None
            or len(result.ids) != len(result.metadatas)
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_collection_page")
        return result

    def safe_get_records(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include_documents: bool = False,
        include_metadata: bool = True,
    ) -> ChromaTransportRecords:
        """Return one bounded business-read page through an approved read handle."""

        if self.requested_lifecycle is not ChromaAccessLifecycle.READ:
            raise ChromaCollectionAuthorityMismatch("chroma_read_lifecycle_required")
        include = tuple(
            name
            for name, enabled in (
                ("documents", include_documents),
                ("metadatas", include_metadata),
            )
            if enabled
        )
        result = self._collection_accessor.get(
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=include,
        )
        if (
            not isinstance(result, ChromaTransportRecords)
            or result.semantic_collection_id != self.semantic_collection_id
            or frozenset(result.included) != frozenset(include)
            or len(result.included) != len(include)
            or (include_documents != (result.documents is not None))
            or (include_metadata != (result.metadatas is not None))
            or (result.documents is not None and len(result.ids) != len(result.documents))
            or (result.metadatas is not None and len(result.ids) != len(result.metadatas))
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_business_read")
        return result

    def safe_vector_query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
        include_documents: bool = False,
        include_metadata: bool = True,
        include_distances: bool = True,
    ) -> ChromaTransportQueryResult:
        """Run one bounded vector query through an approved vector lifecycle."""

        if self.requested_lifecycle is not ChromaAccessLifecycle.VECTOR_QUERY:
            raise ChromaCollectionAuthorityMismatch("chroma_vector_lifecycle_required")
        include = tuple(
            name
            for name, enabled in (
                ("distances", include_distances),
                ("documents", include_documents),
                ("metadatas", include_metadata),
            )
            if enabled
        )
        result = self._collection_accessor.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            ids=ids,
            where=where,
            include=include,
        )
        if (
            not isinstance(result, ChromaTransportQueryResult)
            or result.semantic_collection_id != self.semantic_collection_id
            or frozenset(result.included) != frozenset(include)
            or len(result.included) != len(include)
            or len(result.ids) != 1
            or (include_distances != (result.distances is not None))
            or (include_documents != (result.documents is not None))
            or (include_metadata != (result.metadatas is not None))
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_vector_read")
        expected = len(result.ids[0])
        if (
            result.distances is not None and len(result.distances[0]) != expected
            or result.documents is not None and len(result.documents[0]) != expected
            or result.metadatas is not None and len(result.metadatas[0]) != expected
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_vector_read")
        return result

    def safe_upsert_records(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> ChromaTransportMutationResult:
        """Apply one bounded index upsert through the validated collection accessor."""

        if self.requested_lifecycle is not ChromaAccessLifecycle.INDEX:
            raise ChromaCollectionAuthorityMismatch("chroma_index_lifecycle_required")
        result = self._collection_accessor.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        if (
            not isinstance(result, ChromaTransportMutationResult)
            or result.semantic_collection_id != self.semantic_collection_id
            or result.operation != "upsert"
            or result.requested_count != len(ids)
            or result.affected_count != len(ids)
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_upsert_result")
        return result

    def safe_delete_records(self, *, ids: Sequence[str]) -> ChromaTransportMutationResult:
        """Apply one bounded ID delete through an approved write or index lifecycle."""

        if self.requested_lifecycle not in {
            ChromaAccessLifecycle.WRITE,
            ChromaAccessLifecycle.INDEX,
        }:
            raise ChromaCollectionAuthorityMismatch("chroma_mutation_lifecycle_required")
        result = self._collection_accessor.delete(ids=ids)
        if (
            not isinstance(result, ChromaTransportMutationResult)
            or result.semantic_collection_id != self.semantic_collection_id
            or result.operation != "delete"
            or result.requested_count != len(ids)
            or not 0 <= result.affected_count <= len(ids)
        ):
            raise ChromaCollectionAuthorityMismatch("invalid_chroma_delete_result")
        return result

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaCollectionHandle("
            f"semantic_collection_id={summary['semantic_collection_id']!r}, "
            f"requested_lifecycle={summary['requested_lifecycle']!r}, "
            f"consumer_id={summary['consumer_id']!r}, "
            f"collection_bound={summary['collection_bound']!r})"
        )


TransportConstructor = Callable[..., Any]


def _default_bounded_transport_constructor(
    *, config: ChromaDeploymentConfig
) -> BoundedChromaHttpTransport:
    """Construct the sole production HTTP transport from validated configuration."""

    return BoundedChromaHttpTransport(config)


def _is_unavailable_transport_error(error: BaseException) -> bool:
    return isinstance(error, (ConnectionError, TimeoutError, OSError)) or (
        type(error).__name__ in _UNAVAILABLE_TRANSPORT_ERROR_NAMES
    )


def _parse_lifecycle(value: ChromaAccessLifecycle | str) -> ChromaAccessLifecycle:
    if isinstance(value, ChromaAccessLifecycle):
        return value
    if not isinstance(value, str):
        raise UnsupportedChromaLifecycle("unsupported_chroma_lifecycle")
    try:
        return ChromaAccessLifecycle(value)
    except ValueError:
        raise UnsupportedChromaLifecycle("unsupported_chroma_lifecycle") from None


def _definition(semantic_collection_id: str) -> ChromaCollectionDefinition:
    try:
        return get_collection_definition(semantic_collection_id)
    except UnknownCollectionSemanticId:
        raise UnknownChromaCollection("unknown_chroma_collection") from None


def _legacy_consumer(
    definition: ChromaCollectionDefinition,
    consumer_id: str,
) -> LegacyCollectionConsumer | None:
    matches = [
        consumer
        for consumer in definition.legacy_consumers
        if consumer.consumer_id == consumer_id
    ]
    return matches[0] if len(matches) == 1 else None


def _known_consumer_ids(definition: ChromaCollectionDefinition) -> frozenset[str]:
    approved = {
        consumer
        for consumers in definition.approved_consumers.to_dict().values()
        for consumer in consumers
    }
    return frozenset(
        approved | {consumer.consumer_id for consumer in definition.legacy_consumers}
    )


class ChromaHttpClientFactory:
    """Own one lazy bounded transport and gate collections through registry semantics."""

    __slots__ = ("_config", "_test_context", "_transport", "_transport_constructor")

    def __init__(
        self,
        config: ChromaDeploymentConfig,
        transport_constructor: TransportConstructor | None = None,
        *,
        test_context: bool = False,
    ) -> None:
        if not isinstance(config, ChromaDeploymentConfig):
            raise InvalidChromaFactoryConfiguration("invalid_chroma_factory_configuration")
        if not isinstance(config.mode, ChromaDeploymentMode):
            raise InvalidChromaFactoryConfiguration("invalid_chroma_deployment_mode")
        if not config.is_disabled and (
            not config.uses_http or config.host is None or config.port is None
        ):
            raise InvalidChromaFactoryConfiguration("invalid_http_deployment_configuration")
        if transport_constructor is not None and not callable(transport_constructor):
            raise InvalidChromaFactoryConfiguration("invalid_http_transport_constructor")
        if not isinstance(test_context, bool):
            raise InvalidChromaFactoryConfiguration("invalid_chroma_test_context")
        self._config = config
        self._transport_constructor = (
            transport_constructor or _default_bounded_transport_constructor
        )
        self._test_context = test_context
        self._transport: Any | None = None

    @property
    def state(self) -> ChromaFactoryState:
        if self._config.is_disabled:
            return ChromaFactoryState.DISABLED
        if self._transport is None:
            return ChromaFactoryState.UNINITIALIZED
        return ChromaFactoryState.READY

    def get_factory_summary(self) -> dict[str, str | int | bool]:
        return {
            "factory_state": self.state.value,
            "deployment_mode": self._config.mode.value,
            "transport": "http" if self._config.uses_http else "none",
            "registered_collection_count": len(list_registered_collections()),
            "client_cached": self._transport is not None,
            "timeout_enforced": self._config.uses_http,
        }

    def __repr__(self) -> str:
        summary = self.get_factory_summary()
        return (
            "ChromaHttpClientFactory("
            f"factory_state={summary['factory_state']!r}, "
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"transport={summary['transport']!r}, "
            f"registered_collection_count={summary['registered_collection_count']!r}, "
            f"client_cached={summary['client_cached']!r})"
        )

    def _require_http_enabled(self) -> None:
        if self._config.is_disabled:
            raise ChromaFactoryDisabled("chroma_factory_disabled")
        if not self._config.uses_http:
            raise InvalidChromaFactoryConfiguration("unsupported_chroma_transport")

    def get_transport(self) -> Any:
        """Return the instance-cached bounded transport, constructing it only on demand."""

        self._require_http_enabled()
        if self._transport is not None:
            return self._transport
        try:
            transport = self._transport_constructor(config=self._config)
        except (ChromaHttpClientFactoryError, ChromaTransportError):
            raise
        except Exception as error:
            if _is_unavailable_transport_error(error):
                raise ChromaTransportUnavailable("chroma_transport_unavailable") from None
            raise ChromaTransportError("chroma_http_transport_construction_failed") from None
        if transport is None or not callable(getattr(transport, "get_collection", None)):
            raise ChromaTransportProtocolError("invalid_chroma_http_transport")
        self._transport = transport
        return transport

    def get_client(self) -> Any:
        """Compatibility alias for callers that have not yet adopted transport naming."""

        return self.get_transport()

    def close(self) -> None:
        """Close an already-created transport without constructing one."""

        transport = self._transport
        if transport is None:
            return
        try:
            close = getattr(transport, "close", None)
            if callable(close):
                close()
        finally:
            self._transport = None

    def validate_collection_access(
        self,
        semantic_collection_id: str,
        requested_lifecycle: ChromaAccessLifecycle | str,
        consumer_id: str,
        *,
        creation_requested: bool = False,
    ) -> ValidatedChromaCollectionAccess:
        """Validate registry and consumer authority without constructing a client."""

        self._require_http_enabled()
        if creation_requested is not False:
            raise ChromaCollectionCreationNotAllowed("collection_creation_not_allowed")
        lifecycle = _parse_lifecycle(requested_lifecycle)
        definition = _definition(semantic_collection_id)
        if not isinstance(consumer_id, str) or not consumer_id:
            raise UnknownChromaConsumer("unknown_chroma_consumer")

        legacy = _legacy_consumer(definition, consumer_id)
        if legacy is not None and lifecycle.value in legacy.allowed_lifecycles:
            return ValidatedChromaCollectionAccess(
                semantic_collection_id=definition.semantic_id,
                collection_name=definition.collection_name,
                schema_version=definition.schema_version,
                requested_lifecycle=lifecycle,
                consumer_id=consumer_id,
                consumer_authority=ChromaConsumerAuthority.LEGACY_MIGRATION_TARGET,
                creation_allowed=False,
                access_mode=_ACCESS_MODE_BY_LIFECYCLE[lifecycle],
                access_permitted=False,
                migration_target=legacy.migration_target,
            )

        if lifecycle.value not in definition.allowed_lifecycles:
            raise UnsupportedChromaLifecycle("collection_lifecycle_not_permitted")
        approved = definition.approved_consumers.for_lifecycle(lifecycle.value)
        if consumer_id in approved:
            if lifecycle is ChromaAccessLifecycle.TEST_ONLY:
                if not (
                    self._test_context
                    and self._config.mode is ChromaDeploymentMode.EPHEMERAL_TEST
                ):
                    raise ChromaConsumerNotApproved("test_consumer_context_required")
                authority = ChromaConsumerAuthority.TEST_ONLY
            else:
                authority = ChromaConsumerAuthority.APPROVED
            return ValidatedChromaCollectionAccess(
                semantic_collection_id=definition.semantic_id,
                collection_name=definition.collection_name,
                schema_version=definition.schema_version,
                requested_lifecycle=lifecycle,
                consumer_id=consumer_id,
                consumer_authority=authority,
                creation_allowed=False,
                access_mode=_ACCESS_MODE_BY_LIFECYCLE[lifecycle],
                access_permitted=True,
                migration_target=None,
            )
        if consumer_id in _known_consumer_ids(definition):
            raise ChromaConsumerNotApproved("chroma_consumer_not_approved")
        raise UnknownChromaConsumer("unknown_chroma_consumer")

    def get_collection_handle(
        self,
        semantic_collection_id: str,
        requested_lifecycle: ChromaAccessLifecycle | str,
        consumer_id: str,
        *,
        creation_requested: bool = False,
    ) -> ChromaCollectionHandle:
        """Open one existing collection after semantic access validation."""

        access = self.validate_collection_access(
            semantic_collection_id,
            requested_lifecycle,
            consumer_id,
            creation_requested=creation_requested,
        )
        if not access.access_permitted:
            raise ChromaLegacyConsumerRequiresMigration(
                "legacy_chroma_consumer_requires_migration"
            )
        transport = self.get_transport()
        try:
            collection = transport.get_collection(access.semantic_collection_id)
        except ChromaCollectionMissing:
            raise
        except ChromaTransportError:
            raise
        except Exception as error:
            if _is_unavailable_transport_error(error):
                raise ChromaTransportUnavailable("chroma_transport_unavailable") from None
            raise ChromaTransportError("chroma_collection_lookup_failed") from None
        if (
            collection is None
            or getattr(collection, "name", None) != access.collection_name
            or getattr(collection, "semantic_collection_id", None)
            != access.semantic_collection_id
            or getattr(collection, "schema_version", None) != access.schema_version
        ):
            raise ChromaCollectionAuthorityMismatch("chroma_collection_authority_mismatch")
        return ChromaCollectionHandle(access=access, _collection_accessor=collection)
