"""Bounded HTTP-only access to the existing GitHub evidence vector collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypedDict

try:
    from backend.chroma_collection_registry import (
        GITHUB_EVIDENCE_COLLECTION_NAME,
        GITHUB_EVIDENCE_SEMANTIC_ID,
    )
    from backend.chroma_config import load_chroma_deployment_config
    from backend.chroma_read_client import ChromaReadClient
    from backend.project_repository_identity import (
        authority_to_repository_mapping,
        load_project_repository_identity_authority,
        normalize_project_id,
        normalize_repository_identity,
    )
except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch mode
    from chroma_collection_registry import (
        GITHUB_EVIDENCE_COLLECTION_NAME,
        GITHUB_EVIDENCE_SEMANTIC_ID,
    )
    from chroma_config import load_chroma_deployment_config
    from chroma_read_client import ChromaReadClient
    from project_repository_identity import (
        authority_to_repository_mapping,
        load_project_repository_identity_authority,
        normalize_project_id,
        normalize_repository_identity,
    )


VECTOR_QUERY_BACKEND_ENV = "GITHUB_EVIDENCE_VECTOR_QUERY_BACKEND"
CHROMA_HTTP_HOST_ENV = "CHROMA_HTTP_HOST"
CHROMA_HTTP_PORT_ENV = "CHROMA_HTTP_PORT"
CHROMA_HTTP_SSL_ENV = "CHROMA_HTTP_SSL"
CHROMA_HTTP_TIMEOUT_ENV = "CHROMA_HTTP_TIMEOUT_SECONDS"

DISABLED_VECTOR_QUERY_BACKEND = "disabled"
CHROMA_HTTP_VECTOR_QUERY_BACKEND = "chroma_http"
GITHUB_EVIDENCE_COLLECTION = GITHUB_EVIDENCE_COLLECTION_NAME
EXPECTED_EMBEDDING_DIMENSIONS = 384
EXPECTED_DISTANCE_METRIC = "cosine"
DEFAULT_CHROMA_HTTP_HOST = "127.0.0.1"
DEFAULT_CHROMA_HTTP_PORT = 8100
DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS = 5.0
MAX_CHROMA_HTTP_TIMEOUT_SECONDS = 30.0
MAX_VECTOR_TOP_K = 20
MAX_QUERY_CHARS = 180
MAX_FINGERPRINT_RECORDS = 10_000

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_UNSAFE_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*=|"
    r"access[_-]?token\s*=|password\s*=|secret\s*=|credential\s*=)"
)
_SAFE_RESULT_METADATA_KEYS = (
    "chunk_type",
    "commit_sha",
    "github_repository",
    "path",
    "project_id",
    "project_name",
    "repo",
    "repository",
    "repository_project_id",
    "repository_url",
    "source_id",
    "source_type",
)
_FINGERPRINT_METADATA_KEYS = (
    "github_repository",
    "project_id",
    "project_name",
    "repo",
    "repository",
    "repository_project_id",
    "repository_url",
)


class VectorQueryBackendConfig(TypedDict):
    backend: str
    host: str
    port: int
    ssl: bool
    timeout_seconds: float


class LogicalCollectionFingerprint(TypedDict):
    status: str
    collection_name: str
    record_count: int
    record_ids: list[str]
    repositories: list[str]
    fingerprint: str
    error: str


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _parse_bool(value: Any, default: bool) -> bool | None:
    if value is None or value == "":
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_port(value: Any) -> int | None:
    if value is None or value == "":
        return DEFAULT_CHROMA_HTTP_PORT
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65_535 else None


def _parse_timeout(value: Any) -> float | None:
    if value is None or value == "":
        return DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS
    try:
        timeout = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    return min(timeout, MAX_CHROMA_HTTP_TIMEOUT_SECONDS)


def get_vector_query_backend_config(
    environ: Mapping[str, str] | None = None,
) -> VectorQueryBackendConfig:
    """Return bounded local-only configuration; invalid inputs disable the backend."""

    values = _environment(environ)
    backend = str(values.get(VECTOR_QUERY_BACKEND_ENV, "")).strip().casefold()
    if backend not in {DISABLED_VECTOR_QUERY_BACKEND, CHROMA_HTTP_VECTOR_QUERY_BACKEND}:
        backend = DISABLED_VECTOR_QUERY_BACKEND
    host = str(values.get(CHROMA_HTTP_HOST_ENV, DEFAULT_CHROMA_HTTP_HOST)).strip()
    port = _parse_port(values.get(CHROMA_HTTP_PORT_ENV))
    ssl = _parse_bool(values.get(CHROMA_HTTP_SSL_ENV), False)
    timeout = _parse_timeout(values.get(CHROMA_HTTP_TIMEOUT_ENV))
    if host != DEFAULT_CHROMA_HTTP_HOST or port is None or ssl is None or timeout is None:
        backend = DISABLED_VECTOR_QUERY_BACKEND
    return {
        "backend": backend,
        "host": host if host == DEFAULT_CHROMA_HTTP_HOST else DEFAULT_CHROMA_HTTP_HOST,
        "port": port if port is not None else DEFAULT_CHROMA_HTTP_PORT,
        "ssl": ssl if ssl is not None else False,
        "timeout_seconds": timeout if timeout is not None else DEFAULT_CHROMA_HTTP_TIMEOUT_SECONDS,
    }


def is_chroma_http_vector_query_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return get_vector_query_backend_config(environ)["backend"] == CHROMA_HTTP_VECTOR_QUERY_BACKEND


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _UNSAFE_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _safe_metadata(value: Any, *, fingerprint_only: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    allowed = _FINGERPRINT_METADATA_KEYS if fingerprint_only else _SAFE_RESULT_METADATA_KEYS
    result: dict[str, str] = {}
    for key in allowed:
        safe = _safe_string(value.get(key), 300)
        if safe:
            result[key] = safe
    return result


def _load_authority_mapping(authority: Any = None) -> dict[str, Any]:
    candidate = authority if authority is not None else load_project_repository_identity_authority()
    return authority_to_repository_mapping(candidate)


def _canonical_repository(metadata: Mapping[str, Any], mapping: Mapping[str, Any]) -> str:
    raw = (
        metadata.get("repository")
        or metadata.get("repo")
        or metadata.get("github_repository")
        or metadata.get("repository_url")
    )
    repository = normalize_repository_identity(raw)
    if repository in mapping.get("repository_to_project", {}):
        return repository
    alias_target = mapping.get("alias_to_repository", {}).get(repository)
    return alias_target if isinstance(alias_target, str) else ""


def _authorized_metadata(
    metadata: Any,
    *,
    requested_project_id: str,
    authority_mapping: Mapping[str, Any],
    fingerprint_only: bool = False,
) -> dict[str, str] | None:
    safe = _safe_metadata(metadata, fingerprint_only=fingerprint_only)
    repository = _canonical_repository(safe, authority_mapping)
    mapped_project = authority_mapping.get("repository_to_project", {}).get(repository)
    if not repository or not isinstance(mapped_project, str):
        return None
    explicit_project = normalize_project_id(
        safe.get("project_id") or safe.get("repository_project_id") or safe.get("project_name")
    )
    if explicit_project and explicit_project != mapped_project:
        return None
    if requested_project_id and mapped_project != requested_project_id:
        return None
    safe["project_id"] = mapped_project
    safe["repository"] = repository
    safe.pop("repo", None)
    safe.pop("github_repository", None)
    safe.pop("repository_url", None)
    return dict(sorted(safe.items()))


def _semantic_reader(
    *,
    environ: Mapping[str, str] | None,
    read_client: Any | None,
    client_factory: Callable[..., Any] | None,
) -> Any:
    if read_client is not None:
        return read_client
    config_provider = lambda: load_chroma_deployment_config(environ)
    if client_factory is None:
        return ChromaReadClient(config_provider=config_provider)
    return ChromaReadClient(
        config_provider=config_provider,
        factory_builder=client_factory,
    )


def _query_embedding(embedder: Any, query: str) -> list[float] | None:
    embed = getattr(embedder, "embed", None)
    if not callable(embed):
        return None
    try:
        vector = embed(query)
    except Exception:
        return None
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        return None
    if len(vector) != EXPECTED_EMBEDDING_DIMENSIONS:
        return None
    result: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def search_github_evidence_vectors_http(
    *,
    query: Any,
    n_results: Any,
    project_id: Any,
    embedder: Any,
    authority: Any = None,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
    read_client: Any | None = None,
    skip_socket_preflight: bool = False,
) -> list[dict[str, Any]]:
    """Query the exact server-owned collection and return bounded safe records."""

    config = get_vector_query_backend_config(environ)
    safe_query = _safe_string(query, MAX_QUERY_CHARS)
    requested_project = normalize_project_id(project_id)
    if (
        config["backend"] != CHROMA_HTTP_VECTOR_QUERY_BACKEND
        or not safe_query
        or not requested_project
        or isinstance(n_results, bool)
        or not isinstance(n_results, int)
        or n_results <= 0
    ):
        return []
    limit = min(n_results, MAX_VECTOR_TOP_K)
    embedding = _query_embedding(embedder, safe_query)
    authority_mapping = _load_authority_mapping(authority)
    if embedding is None or not authority_mapping.get("mapping_count"):
        return []
    try:
        reader = _semantic_reader(
            environ=environ,
            read_client=read_client,
            client_factory=client_factory,
        )
        result = reader.vector_query(
            GITHUB_EVIDENCE_SEMANTIC_ID,
            consumer_id="github_evidence_vector_reader",
            query_embedding=embedding,
            n_results=limit,
            include_documents=False,
            metadata_fields=_SAFE_RESULT_METADATA_KEYS,
            include_distances=True,
        )
        records: list[dict[str, Any]] = []
        for hit in result.hits[:limit]:
            safe_id = _safe_string(hit.record_id, 180)
            distance = hit.distance
            if (
                not safe_id
                or isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(float(distance))
                or float(distance) < 0
            ):
                continue
            safe_metadata = _authorized_metadata(
                hit.metadata,
                requested_project_id=requested_project,
                authority_mapping=authority_mapping,
            )
            if safe_metadata is None:
                continue
            records.append({
                "vector_record_id": safe_id,
                "distance": float(distance),
                "metadata": safe_metadata,
                "rank": hit.rank,
            })
        return records[:limit]
    except Exception:
        return []


def inspect_github_evidence_vector_metadata_http(
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
    read_client: Any | None = None,
    skip_socket_preflight: bool = False,
) -> list[dict[str, Any]]:
    """Read bounded identity metadata for the existing collection readiness gate.

    Records are deliberately not authority-filtered here: the readiness authority
    must see missing or conflicting identities and fail closed.  Documents and
    embeddings are never requested or returned.
    """

    config = get_vector_query_backend_config(environ)
    if config["backend"] != CHROMA_HTTP_VECTOR_QUERY_BACKEND:
        return []
    try:
        reader = _semantic_reader(
            environ=environ,
            read_client=read_client,
            client_factory=client_factory,
        )
        result = reader.read_records(
            GITHUB_EVIDENCE_SEMANTIC_ID,
            consumer_id="github_evidence_metadata_reader",
            include_documents=False,
            metadata_fields=_FINGERPRINT_METADATA_KEYS,
            max_records=MAX_FINGERPRINT_RECORDS,
        )
        records: list[dict[str, Any]] = []
        for record in result.records:
            safe_id = _safe_string(record.record_id, 180)
            if not safe_id:
                return []
            records.append({
                "vector_record_id": safe_id,
                "metadata": _safe_metadata(record.metadata, fingerprint_only=True),
            })
        return records
    except Exception:
        return []


def compute_github_evidence_logical_fingerprint_http(
    *,
    authority: Any = None,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
    read_client: Any | None = None,
    skip_socket_preflight: bool = False,
) -> LogicalCollectionFingerprint:
    """Hash collection identity, record IDs, and authoritative repository metadata."""

    empty: LogicalCollectionFingerprint = {
        "status": "blocked",
        "collection_name": GITHUB_EVIDENCE_COLLECTION,
        "record_count": 0,
        "record_ids": [],
        "repositories": [],
        "fingerprint": "",
        "error": "logical_fingerprint_unavailable",
    }
    config = get_vector_query_backend_config(environ)
    if config["backend"] != CHROMA_HTTP_VECTOR_QUERY_BACKEND:
        return empty
    authority_mapping = _load_authority_mapping(authority)
    if not authority_mapping.get("mapping_count"):
        return empty
    try:
        reader = _semantic_reader(
            environ=environ,
            read_client=read_client,
            client_factory=client_factory,
        )
        result = reader.read_records(
            GITHUB_EVIDENCE_SEMANTIC_ID,
            consumer_id="github_evidence_metadata_reader",
            include_documents=False,
            metadata_fields=_FINGERPRINT_METADATA_KEYS,
            max_records=MAX_FINGERPRINT_RECORDS,
        )
        records = []
        for record in result.records:
            safe_id = _safe_string(record.record_id, 180)
            safe_metadata = _authorized_metadata(
                record.metadata,
                requested_project_id="",
                authority_mapping=authority_mapping,
                fingerprint_only=True,
            )
            if not safe_id or safe_metadata is None:
                return empty
            records.append({"id": safe_id, "metadata": safe_metadata})
        records.sort(key=lambda item: item["id"])
        payload = {
            "collection_name": GITHUB_EVIDENCE_COLLECTION,
            "record_count": len(records),
            "records": records,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        repositories = sorted({item["metadata"]["repository"] for item in records})
        return {
            "status": "ready",
            "collection_name": GITHUB_EVIDENCE_COLLECTION,
            "record_count": len(records),
            "record_ids": [item["id"] for item in records],
            "repositories": repositories,
            "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "error": "",
        }
    except Exception:
        return empty
