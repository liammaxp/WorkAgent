"""Safe multi-query vector retrieval and repository-to-chunk mapping."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping, Sequence, TypedDict

from backend import evidence_memory
from backend.evidence_chunk_search import keyword_search_evidence_chunks
from backend.project_query_planner import QUERY_GROUPS


MAX_VECTOR_QUERY_CHARS = 180
MAX_VECTOR_QUERY_TERMS = 16
MAX_VECTOR_QUERIES = 12
DEFAULT_VECTOR_TOP_K_PER_QUERY = 5
MAX_VECTOR_TOP_K_PER_QUERY = 20
DEFAULT_CHUNKS_PER_VECTOR_RESULT = 3
MAX_CHUNKS_PER_VECTOR_RESULT = 10
MAX_VECTOR_RESULTS_TOTAL = 100
MAX_VECTOR_HITS_TOTAL = 120
MAX_VECTOR_MATCH_REASONS = 16
MAX_VECTOR_MATCHED_TERMS = 16
MAX_VECTOR_HIT_SUMMARY_CHARS = 500
MAPPING_SCORE_CAP = 40.0
VECTOR_SCORE_WEIGHT = 0.65
MAPPING_SCORE_WEIGHT = 0.35

ELIGIBLE_VECTOR_QUERY_GROUPS = (
    "project_identity",
    "jd_alignment",
    "mechanisms",
    "validation_repair",
    "metrics_impact",
)
_GROUP_ORDER = {group: index for index, group in enumerate(QUERY_GROUPS)}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,127}")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+.#/-]{1,160}$")
_UNSAFE_RE = re.compile(
    r"(?i)(?:diff\s+--git|begin\s+(?:rsa\s+)?private\s+key|api[_-]?key\s*=|"
    r"access[_-]?token\s*=|password\s*=|secret\s*=|credential\s*=)"
)
_SAFE_METADATA_KEYS = frozenset({
    "chunk_type", "commit_sha", "path", "project_id", "project_name", "repo",
    "repository", "repository_project_id", "source_id", "source_type",
})


class NormalizedRepositoryVectorResult(TypedDict):
    vector_record_id: str
    project_id: str
    repo: str
    path: str
    source_type: str
    vector_score: float
    distance: float | None
    vector_rank: int
    safe_metadata: dict[str, str]


class VectorEvidenceHit(TypedDict):
    hit_id: str
    vector_record_id: str
    vector_rank: int
    chunk_id: str
    source_id: str
    project_id: str
    repo: str
    path: str
    commit_sha: str
    source_type: str
    chunk_type: str
    symbol: str
    search_source: str
    query_group: str
    query: str
    vector_score: float
    mapping_score: float
    score: float
    matched_terms: list[str]
    match_reasons: list[str]
    summary: str
    keywords: list[str]
    technical_tags: list[str]
    text_hash: str
    text_chars: int


def _safe_string(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(_CONTROL_RE.sub(" ", value).split())
    if not cleaned or _UNSAFE_RE.search(cleaned):
        return ""
    return cleaned[:limit]


def _safe_query(value: Any) -> str:
    safe = _safe_string(value, MAX_VECTOR_QUERY_CHARS * 4)
    if not safe:
        return ""
    terms = sorted({
        token.strip("/.-").casefold()
        for token in _TOKEN_RE.findall(safe)
        if token.strip("/.-")
    })[:MAX_VECTOR_QUERY_TERMS]
    if not terms:
        return ""
    return " ".join(terms)[:MAX_VECTOR_QUERY_CHARS]


def _positive_limit(value: Any, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return 0
    return min(value, maximum)


def normalize_vector_score(*, distance: Any = None, similarity: Any = None) -> float | None:
    """Normalize similarity directly, or non-negative distance as ``1 / (1 + distance)``."""

    if similarity is not None:
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            return None
        value = float(similarity)
        if not math.isfinite(value) or value < 0:
            return None
        return round(min(value, 1.0), 6)
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return None
    value = float(distance)
    if not math.isfinite(value) or value < 0:
        return None
    return round(1.0 / (1.0 + value), 6)


def _safe_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key in sorted(value, key=lambda item: str(item).casefold()):
        normalized_key = str(key).casefold()
        if normalized_key not in _SAFE_METADATA_KEYS:
            continue
        safe = _safe_string(value[key], 300)
        if safe:
            result[normalized_key] = safe
    return result


def normalize_repository_vector_result(
    result: Any,
    *,
    project_id: Any,
    vector_rank: int = 0,
) -> NormalizedRepositoryVectorResult | None:
    if not isinstance(result, Mapping):
        return None
    requested_project = _safe_string(project_id, 160)
    if not requested_project:
        return None
    metadata = _safe_metadata(result.get("metadata"))
    for key in _SAFE_METADATA_KEYS:
        if key not in metadata:
            safe = _safe_string(result.get(key), 300)
            if safe:
                metadata[key] = safe
    result_project = metadata.get("project_id") or metadata.get("project_name") or ""
    if not result_project or result_project.casefold() != requested_project.casefold():
        return None
    repository_project = metadata.get("repository_project_id")
    if repository_project and repository_project.casefold() != requested_project.casefold():
        return None
    score = normalize_vector_score(
        distance=result.get("distance"), similarity=result.get("similarity", result.get("vector_score"))
    )
    if score is None:
        return None
    distance_value = result.get("distance")
    distance = (
        round(float(distance_value), 6)
        if isinstance(distance_value, (int, float))
        and not isinstance(distance_value, bool)
        and math.isfinite(float(distance_value))
        and float(distance_value) >= 0
        else None
    )
    record_id = _safe_string(
        result.get("vector_record_id") or result.get("id"), 180
    )
    if not record_id:
        return None
    explicit_rank = result.get("rank", result.get("vector_rank"))
    rank = explicit_rank if isinstance(explicit_rank, int) and not isinstance(explicit_rank, bool) and explicit_rank > 0 else vector_rank
    if rank <= 0:
        rank = 1
    return {
        "vector_record_id": record_id,
        "project_id": result_project,
        "repo": metadata.get("repo") or metadata.get("repository") or "",
        "path": metadata.get("path", ""),
        "source_type": metadata.get("source_type", ""),
        "vector_score": score,
        "distance": distance,
        "vector_rank": rank,
        "safe_metadata": metadata,
    }


def search_existing_github_evidence_vectors(
    *,
    query: Any,
    n_results: Any = DEFAULT_VECTOR_TOP_K_PER_QUERY,
    project_id: Any = None,
    memory_store: Any = None,
) -> list[dict[str, Any]]:
    """Thin read-only adapter over the existing GitHub vector collection."""

    safe_query = _safe_query(query)
    limit = _positive_limit(n_results, MAX_VECTOR_TOP_K_PER_QUERY)
    if not safe_query or not limit or memory_store is None:
        return []
    search = getattr(memory_store, "search_github_vector_records", None)
    if not callable(search):
        return []
    requested_project = _safe_string(project_id, 160)
    if not requested_project:
        return []
    try:
        results = search(safe_query, n_results=limit, project_id=requested_project)
    except Exception:
        return []
    if not isinstance(results, list):
        return []
    normalized = [
        item
        for rank, result in enumerate(results[:limit], start=1)
        if (item := normalize_repository_vector_result(
            result, project_id=requested_project, vector_rank=rank
        )) is not None
    ]
    return sorted(normalized, key=lambda item: (-item["vector_score"], item["vector_record_id"]))


def _repo_equal(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        return value.strip().replace("\\", "/").removesuffix(".git").strip("/").casefold()
    return bool(left and right and normalize(left) == normalize(right))


def _path_equal(left: str, right: str) -> bool:
    return bool(left and right and left.replace("\\", "/").strip("/").casefold() == right.replace("\\", "/").strip("/").casefold())


def map_vector_result_to_evidence_chunks(
    *,
    vector_result: NormalizedRepositoryVectorResult,
    query: Any,
    query_group: Any,
    project_id: Any,
    chunks: Any,
    top_k: Any = DEFAULT_CHUNKS_PER_VECTOR_RESULT,
) -> list[VectorEvidenceHit]:
    safe_query = _safe_query(query)
    requested_project = _safe_string(project_id, 160)
    limit = _positive_limit(top_k, MAX_CHUNKS_PER_VECTOR_RESULT)
    if not safe_query or not requested_project or not limit or not isinstance(chunks, (list, tuple)):
        return []
    if vector_result["project_id"].casefold() != requested_project.casefold():
        return []
    candidates = [
        item for item in chunks
        if isinstance(item, Mapping)
        and isinstance(item.get("project_id"), str)
        and item["project_id"].strip().casefold() == requested_project.casefold()
    ]
    repo = vector_result["repo"]
    if repo:
        repo_candidates = [
            item for item in candidates
            if isinstance(item.get("repo"), str) and _repo_equal(item["repo"], repo)
        ]
        if not repo_candidates:
            return []
        candidates = repo_candidates
    path = vector_result["path"]
    if path:
        path_candidates = [
            item for item in candidates
            if isinstance(item.get("path"), str) and _path_equal(item["path"], path)
        ]
        if path_candidates:
            candidates = path_candidates
    lexical_hits = keyword_search_evidence_chunks(
        chunks=candidates,
        query=safe_query,
        project_id=requested_project,
        query_group=_safe_string(query_group, 64),
        top_k=limit,
    )
    hits: list[VectorEvidenceHit] = []
    for lexical in lexical_hits:
        repo_bonus = 2.0 if repo and _repo_equal(lexical["repo"], repo) else 0.0
        path_bonus = 8.0 if path and _path_equal(lexical["path"], path) else 0.0
        mapping_score = round(min(lexical["score"] + repo_bonus + path_bonus, MAPPING_SCORE_CAP), 4)
        normalized_mapping = min(mapping_score / MAPPING_SCORE_CAP, 1.0)
        final_score = round(
            vector_result["vector_score"] * VECTOR_SCORE_WEIGHT
            + normalized_mapping * MAPPING_SCORE_WEIGHT,
            6,
        )
        reasons = ["vector_repository_match"]
        if repo_bonus:
            reasons.append("repository_exact")
        if path_bonus:
            reasons.append("path_exact")
        reasons.extend(lexical["match_reasons"])
        reasons = sorted(set(reasons))[:MAX_VECTOR_MATCH_REASONS]
        matched = lexical["matched_terms"][:MAX_VECTOR_MATCHED_TERMS]
        hit_id = evidence_memory.stable_record_id(
            "vhit",
            [
                vector_result["vector_record_id"], lexical["chunk_id"],
                _safe_string(query_group, 64), safe_query,
            ],
        )
        hits.append({
            "hit_id": hit_id,
            "vector_record_id": vector_result["vector_record_id"],
            "vector_rank": vector_result["vector_rank"],
            "chunk_id": lexical["chunk_id"],
            "source_id": lexical["source_id"],
            "project_id": lexical["project_id"],
            "repo": lexical["repo"],
            "path": lexical["path"],
            "commit_sha": lexical["commit_sha"],
            "source_type": lexical["source_type"],
            "chunk_type": lexical["chunk_type"],
            "symbol": lexical["symbol"],
            "search_source": "vector",
            "query_group": _safe_string(query_group, 64),
            "query": safe_query,
            "vector_score": vector_result["vector_score"],
            "mapping_score": mapping_score,
            "score": final_score,
            "matched_terms": matched,
            "match_reasons": reasons,
            "summary": lexical["summary"][:MAX_VECTOR_HIT_SUMMARY_CHARS],
            "keywords": lexical["keywords"],
            "technical_tags": lexical["technical_tags"],
            "text_hash": lexical["text_hash"],
            "text_chars": lexical["text_chars"],
        })
    return hits


def _hit_order(hit: VectorEvidenceHit) -> tuple[Any, ...]:
    return (
        -hit["score"], -hit["vector_score"], -hit["mapping_score"],
        _GROUP_ORDER.get(hit["query_group"], len(_GROUP_ORDER)),
        hit["query"].casefold(), hit["path"].casefold(), hit["symbol"].casefold(),
        hit["chunk_id"], hit["hit_id"],
    )


def vector_search_project_query_plan(
    *,
    query_plan: Any,
    project_id: Any,
    chunks: Any,
    vector_search: Callable[..., Any],
    top_k_per_query: Any = DEFAULT_VECTOR_TOP_K_PER_QUERY,
    chunks_per_vector_result: Any = DEFAULT_CHUNKS_PER_VECTOR_RESULT,
) -> list[VectorEvidenceHit]:
    requested_project = _safe_string(project_id, 160)
    vector_limit = _positive_limit(top_k_per_query, MAX_VECTOR_TOP_K_PER_QUERY)
    chunk_limit = _positive_limit(chunks_per_vector_result, MAX_CHUNKS_PER_VECTOR_RESULT)
    if (
        not requested_project or not isinstance(query_plan, Mapping)
        or not isinstance(chunks, (list, tuple)) or not chunks
        or not callable(vector_search) or not vector_limit or not chunk_limit
    ):
        return []
    queries: list[tuple[str, str]] = []
    for group in ELIGIBLE_VECTOR_QUERY_GROUPS:
        values = query_plan.get(group)
        if not isinstance(values, (list, tuple)):
            continue
        normalized = sorted({_safe_query(value) for value in values if _safe_query(value)})
        queries.extend((group, query) for query in normalized)
        if len(queries) >= MAX_VECTOR_QUERIES:
            break
    queries = queries[:MAX_VECTOR_QUERIES]
    hits: list[VectorEvidenceHit] = []
    received = 0
    for group, query in queries:
        try:
            raw_results = vector_search(
                query=query, n_results=vector_limit, project_id=requested_project
            )
        except Exception:
            continue
        if not isinstance(raw_results, (list, tuple)):
            continue
        normalized_results = []
        for raw in raw_results[:vector_limit]:
            normalized = normalize_repository_vector_result(raw, project_id=requested_project)
            if normalized is not None:
                normalized_results.append(normalized)
        normalized_results.sort(
            key=lambda item: (-item["vector_score"], item["vector_record_id"])
        )
        for canonical_rank, result in enumerate(normalized_results, start=1):
            if not (isinstance(raw_results, list) and any(
                isinstance(raw, Mapping) and raw.get("rank") == result["vector_rank"] for raw in raw_results
            )):
                result["vector_rank"] = canonical_rank
            received += 1
            if received > MAX_VECTOR_RESULTS_TOTAL:
                break
            hits.extend(map_vector_result_to_evidence_chunks(
                vector_result=result, query=query, query_group=group,
                project_id=requested_project, chunks=chunks, top_k=chunk_limit,
            ))
            if len(hits) >= MAX_VECTOR_HITS_TOTAL:
                return sorted(hits, key=_hit_order)[:MAX_VECTOR_HITS_TOTAL]
        if received >= MAX_VECTOR_RESULTS_TOTAL:
            break
    return sorted(hits, key=_hit_order)[:MAX_VECTOR_HITS_TOTAL]
