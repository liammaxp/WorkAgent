"""Default-off, read-only integration for project evidence retrieval v2."""

from __future__ import annotations

import os
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.chroma_http_vector_search import (
    inspect_github_evidence_vector_metadata_http,
    is_chroma_http_vector_query_enabled,
    search_github_evidence_vectors_http,
)
from backend.evidence_hybrid_retrieval import run_project_hybrid_retrieval
from backend.evidence_index_readiness import inspect_evidence_index_readiness
from backend.github_evidence_chunks import (
    DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    load_github_evidence_chunk_records,
)
from backend.github_evidence_materializer import (
    DEFAULT_MATERIALIZATION_MANIFEST_PATH,
)
from backend.github_raw_storage import DEFAULT_GITHUB_RAW_SOURCES_PATH
from backend.memory_store import LocalHashEmbedding
from backend.project_evidence_followup_intents import validate_followup_retrieval_intents
from backend.project_query_planner import QUERY_GROUPS, build_project_query_plan
from backend.project_repository_identity import (
    DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
    authority_to_repository_mapping,
    load_project_repository_identity_authority,
    normalize_project_id,
    normalize_repository_identity,
)


GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG = "USE_GITHUB_EVIDENCE_RETRIEVAL_V2"
_ENABLED_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_V2_EVIDENCE_LIMIT = 20
MAX_V2_EVIDENCE_LIMIT = 50
MAX_V2_SAFE_SUMMARY_CHARS = 600
MAX_V2_SAFE_LABELS = 32

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_UNSAFE_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*[:=]|"
    r"access[_-]?token\s*[:=]|password\s*[:=]|credential\s*[:=]|"
    r"full\s+patch|full\s+source\s+body|vector\s+document)"
)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
_SAFE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEARCH_SOURCES = frozenset({"keyword", "symbol", "vector"})


def is_github_evidence_retrieval_v2_enabled() -> bool:
    """Read the flag at call time, defaulting and failing closed."""

    try:
        value = os.getenv(GITHUB_EVIDENCE_RETRIEVAL_V2_FLAG, "")
    except Exception:
        return False
    return isinstance(value, str) and value.strip().casefold() in _ENABLED_FLAG_VALUES


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _UNSAFE_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _safe_labels(value: Any, *, allowed: frozenset[str] | None = None) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    labels: dict[str, str] = {}
    for item in sorted(value, key=lambda candidate: (str(candidate).casefold(), str(candidate)))[
        : MAX_V2_SAFE_LABELS * 4
    ]:
        safe = _safe_string(item, 160)
        if not safe:
            continue
        normalized = safe.casefold()
        if allowed is not None and normalized not in allowed:
            continue
        labels[normalized] = safe
    return [labels[key] for key in sorted(labels)][:MAX_V2_SAFE_LABELS]


def _bounded_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) and 0 <= number <= 1 else None


def _project_memory_for_request(project: Mapping[str, Any]) -> dict[str, Any]:
    return {"projects": [dict(project)]}


def _ready_for_requested_project(readiness: Any, project_id: str) -> bool:
    if not isinstance(readiness, Mapping):
        return False
    projects = readiness.get("projects_with_chunks")
    return bool(
        readiness.get("status") == "ready"
        and readiness.get("vector_ready") is True
        and readiness.get("chunk_mapping_ready") is True
        and readiness.get("ready_for_hybrid_retrieval") is True
        and readiness.get("records_unresolved") == 0
        and readiness.get("identity_authority_status") == "ready"
        and readiness.get("identity_conflict_count") == 0
        and readiness.get("identity_unresolved_count") == 0
        and isinstance(projects, (list, tuple))
        and any(normalize_project_id(item) == project_id for item in projects)
    )


def adapt_hybrid_hits_for_resume_evidence(
    hits: Any,
    *,
    project_id: Any,
    authorized_repositories: Sequence[str],
    project_name: Any = "",
    limit: Any = DEFAULT_V2_EVIDENCE_LIMIT,
) -> list[dict[str, Any]]:
    """Allowlist safe hybrid fields into the existing list-shaped evidence contract."""

    requested_project = normalize_project_id(project_id)
    if (
        not requested_project
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or not isinstance(hits, (list, tuple))
    ):
        return []
    maximum = min(limit, MAX_V2_EVIDENCE_LIMIT)
    authorized = {
        repository
        for value in authorized_repositories
        if (repository := normalize_repository_identity(value))
    }
    if not authorized:
        return []
    safe_project_name = _safe_string(project_name, 160)
    adapted: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for hit in hits:
        if not isinstance(hit, Mapping) or normalize_project_id(hit.get("project_id")) != requested_project:
            continue
        repository = normalize_repository_identity(hit.get("repo"))
        chunk_id = _safe_string(hit.get("chunk_id"), 180)
        source_id = _safe_string(hit.get("source_id"), 180)
        if (
            repository not in authorized
            or not _SAFE_ID_RE.fullmatch(chunk_id)
            or not _SAFE_ID_RE.fullmatch(source_id)
            or chunk_id in seen_chunks
        ):
            continue
        final_score = _bounded_score(hit.get("score"))
        keyword_score = _bounded_score(hit.get("keyword_score"))
        symbol_score = _bounded_score(hit.get("symbol_score"))
        vector_score = _bounded_score(hit.get("vector_score"))
        if None in (final_score, keyword_score, symbol_score, vector_score):
            continue
        path = _safe_string(hit.get("path"), 400)
        summary = _safe_string(hit.get("summary"), MAX_V2_SAFE_SUMMARY_CHARS)
        text_hash = _safe_string(hit.get("text_hash"), 64).casefold()
        text_chars = hit.get("text_chars")
        if not _SAFE_HASH_RE.fullmatch(text_hash):
            text_hash = ""
        if isinstance(text_chars, bool) or not isinstance(text_chars, int) or text_chars < 0:
            text_chars = 0
        keywords = _safe_labels(hit.get("keywords"))
        technical_tags = _safe_labels(hit.get("technical_tags"))
        search_sources = _safe_labels(hit.get("search_sources"), allowed=_SAFE_SEARCH_SOURCES)
        query_groups = _safe_labels(hit.get("query_groups"), allowed=frozenset(QUERY_GROUPS))
        match_reasons = _safe_labels(hit.get("match_reasons"))
        record = {
            "project_id": requested_project,
            "project_name": safe_project_name or requested_project,
            "chunk_id": chunk_id,
            "source_id": source_id,
            "repo": repository,
            "repository": repository,
            "path": path,
            "commit_sha": _safe_string(hit.get("commit_sha"), 180),
            "source_type": _safe_string(hit.get("source_type"), 64),
            "chunk_type": _safe_string(hit.get("chunk_type"), 64),
            "symbol": _safe_string(hit.get("symbol"), 180),
            "summary": summary,
            "description": summary,
            "keywords": keywords,
            "technical_tags": technical_tags,
            "final_score": final_score,
            "keyword_score": keyword_score,
            "symbol_score": symbol_score,
            "vector_score": vector_score,
            "component_scores": {
                "keyword": keyword_score,
                "symbol": symbol_score,
                "vector": vector_score,
            },
            "query_groups": query_groups,
            "search_sources": search_sources,
            "match_reasons": match_reasons,
            "text_hash": text_hash,
            "text_chars": min(text_chars, 2_000_000),
            "root_files": [path] if path else [],
        }
        seen_chunks.add(chunk_id)
        adapted.append(record)
        if len(adapted) >= maximum:
            break
    return adapted


def retrieve_evidence_for_project_v2(
    project: dict[str, Any],
    *,
    jd_targets: Any = None,
    compact_facts: Any = None,
    known_symbols: Any = None,
    retrieval_intents: Any = None,
    limit: Any = DEFAULT_V2_EVIDENCE_LIMIT,
    readiness_inspector: Callable[..., Any] = inspect_evidence_index_readiness,
    vector_metadata_reader: Callable[..., Any] = inspect_github_evidence_vector_metadata_http,
    chunk_loader: Callable[..., Any] = load_github_evidence_chunk_records,
    query_plan_builder: Callable[..., Any] = build_project_query_plan,
    hybrid_retriever: Callable[..., Any] = run_project_hybrid_retrieval,
    vector_search: Callable[..., Any] = search_github_evidence_vectors_http,
    vector_backend_enabled: Callable[..., Any] = is_chroma_http_vector_query_enabled,
    authority_loader: Callable[..., Any] = load_project_repository_identity_authority,
    embedder: Any = None,
    raw_source_path: str | Path = DEFAULT_GITHUB_RAW_SOURCES_PATH,
    chunk_path: str | Path = DEFAULT_GITHUB_EVIDENCE_CHUNKS_PATH,
    manifest_path: str | Path = DEFAULT_MATERIALIZATION_MANIFEST_PATH,
    identity_authority_path: str | Path = DEFAULT_PROJECT_REPOSITORY_IDENTITY_PATH,
) -> list[dict[str, Any]]:
    """Run the accepted retrieval chain without writes, fallback, or raw output."""

    if not isinstance(project, Mapping):
        return []
    project_id = normalize_project_id(project.get("project_id"))
    if (
        not project_id
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        return []
    try:
        validated_intents = validate_followup_retrieval_intents(
            project_id=project.get("project_id"),
            retrieval_intents=retrieval_intents,
        )
    except (TypeError, ValueError):
        return []
    try:
        if vector_backend_enabled() is not True:
            return []
        authority = authority_loader(identity_authority_path)
        mapping = authority_to_repository_mapping(authority)
        authorized_repositories = sorted(
            repository
            for repository, mapped_project in mapping["repository_to_project"].items()
            if mapped_project == project_id
        )
        if (
            not isinstance(authority, Mapping)
            or authority.get("status") != "ready"
            or authority.get("conflicts") != []
            or authority.get("unresolved_projects") != []
            or authority.get("unresolved_repositories") != []
            or not authorized_repositories
        ):
            return []
        vector_records = vector_metadata_reader()
        readiness = readiness_inspector(
            project_memory=_project_memory_for_request(project),
            vector_records=vector_records,
            raw_source_path=raw_source_path,
            chunk_path=chunk_path,
            manifest_path=manifest_path,
            intended_project_ids=[project_id],
            identity_authority=authority,
        )
        if not _ready_for_requested_project(readiness, project_id):
            return []
        chunks = chunk_loader(chunk_path)
        project_chunks = [
            item for item in chunks
            if isinstance(item, Mapping) and normalize_project_id(item.get("project_id")) == project_id
        ] if isinstance(chunks, (list, tuple)) else []
        if not project_chunks:
            return []
        planner_kwargs = {
            "project_id": project_id,
            "project_memory": _project_memory_for_request(project),
            "compact_facts": compact_facts,
            "jd_targets": jd_targets,
            "known_symbols": known_symbols,
        }
        if validated_intents:
            planner_kwargs["retrieval_intents"] = validated_intents
        plan = query_plan_builder(**planner_kwargs)
        if not isinstance(plan, Mapping) or plan.get("project_id") != project_id:
            return []
        local_embedder = embedder if embedder is not None else LocalHashEmbedding()

        def http_vector_search(*, query: Any, n_results: Any, project_id: Any) -> list[dict[str, Any]]:
            return vector_search(
                query=query,
                n_results=n_results,
                project_id=project_id,
                embedder=local_embedder,
                authority=authority,
            )

        result = hybrid_retriever(
            project_id=project_id,
            query_plan=plan,
            chunks=project_chunks,
            vector_search=http_vector_search,
            readiness=readiness,
            top_k=min(limit, MAX_V2_EVIDENCE_LIMIT),
        )
        if not isinstance(result, Mapping) or result.get("status") != "ready":
            return []
        hits = result.get("hits")
        if not isinstance(hits, (list, tuple)) or not any(
            isinstance(hit, Mapping)
            and "vector" in hit.get("search_sources", [])
            for hit in hits
        ):
            return []
        return adapt_hybrid_hits_for_resume_evidence(
            hits,
            project_id=project_id,
            project_name=project.get("project_name") or project.get("name"),
            authorized_repositories=authorized_repositories,
            limit=limit,
        )
    except Exception:
        return []
